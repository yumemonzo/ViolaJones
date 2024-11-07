import os
import pickle
import logging
import numpy as np
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
from weak_classifier import WeakClassifier
from visualizer import plot_performance_metrics, plot_thresholds, plot_alphas, plot_label_length


class AdaBoost:
    def __init__(self, layers):
        self.layers = layers
        self.weak_classifiers = []
        self.alphas = []
        self.train_integral_images = None
        self.train_y = None
        self.test_integral_images = None
        self.test_y = None

    def train(self, train_integral_images, train_labels, test_integral_images, test_labels, haar_like_filters, save_dir):
        self.train_integral_images = train_integral_images
        self.train_y = train_labels
        self.test_integral_images = test_integral_images
        self.test_y = test_labels
        train_integral_images = np.array(train_integral_images)
        train_labels = np.array(train_labels)
        
        train_stage_matrics = []
        test_stage_matrics = []
        train_label_lengths_per_stage = []
        weights = np.ones(train_labels.shape[0]) / train_labels.shape[0]
        
        for stage, n_weak_classifiers in enumerate(self.layers):
            stage_weak_classifiers = []
            stage_alphas = []
            for idx in range(n_weak_classifiers):
                best_weak_classifier = None
                best_alpha = 0
                best_error = float('inf')
                with ProcessPoolExecutor() as executor:
                    args = [(train_integral_images, train_labels, haar_like_filter, weights) for haar_like_filter in haar_like_filters]
                    results = list(tqdm(executor.map(self.train_weak_classifier_wrapper, args), total=len(haar_like_filters), desc=f'Training classifiers for stage {stage}'))

                for weak_classifier, predictions, error in results:
                    if error == 0:
                        alpha = 1e-10
                        stage_weights = weights.copy()
                        stage_weights /= np.sum(stage_weights)
                    elif error == 1:
                        alpha = 1e-10
                        stage_weights = weights.copy()
                        stage_weights /= np.sum(stage_weights)                        
                    else:
                        alpha = 0.5 * np.log((1 - error) / (error + 1e-10))
                        stage_weights = weights.copy()
                        stage_weights *= np.exp(alpha * ((predictions != train_labels) * 2 - 1))
                        stage_weights /= np.sum(stage_weights)

                    if error <= best_error:
                        best_error = error
                        best_alpha = alpha
                        best_weak_classifier = weak_classifier

                if best_weak_classifier is not None:
                    stage_weak_classifiers.append(best_weak_classifier)
                    stage_alphas.append(best_alpha)

            train_metrics = self.evaluate(self.train_integral_images, self.train_y)
            train_stage_matrics.append(train_metrics)
            logging.info(f"[Evaluate] {stage}번째 Train 성능 평가 결과:")
            for key, value in train_metrics.items():
                logging.info(f"- {key}: {value:.2f}")
            logging.info(f"-----------------------------------------------------------------------------------------------")

            test_metrics = self.evaluate(self.test_integral_images, self.test_y)
            test_stage_matrics.append(test_metrics)
            logging.info(f"[Evaluate] {stage}번째 Test 성능 평가 결과:")
            for key, value in test_metrics.items():
                logging.info(f"- {key}: {value:.2f}")
            logging.info(f"-----------------------------------------------------------------------------------------------")

            train_label_lengths_per_stage.append(len(train_labels))
            train_integral_images, train_labels, weights = self.process_predictions(stage, train_integral_images, train_labels)

            self.weak_classifiers.append(stage_weak_classifiers)
            self.alphas.append(stage_alphas)

            plot_performance_metrics(train_stage_matrics, os.path.join(save_dir, 'train_metrics.jpg'))
            plot_performance_metrics(test_stage_matrics, os.path.join(save_dir, 'test_metrics.jpg'))
            plot_thresholds(self.weak_classifiers, os.path.join(save_dir, 'weak_classifier_thresholds.jpg'))
            plot_alphas(self.alphas, os.path.join(save_dir, 'alphas.jpg'))
            plot_label_length(train_label_lengths_per_stage, os.path.join(save_dir, 'train_label_length.jpg'))

            if len(train_labels) == 0:
                logging.info(f"{stage}단계 이후 잘못된 샘플이 남아 있지 않아 학습이 중단되었습니다.")
                logging.info(f"-----------------------------------------------------------------------------------------------")
                break

    def filter_correct_indices(self, total_indices, incorrect_indices):
        """올바르게 예측된 인덱스를 반환합니다."""
        return [i for i in total_indices if i not in incorrect_indices]

    def count_correct_predictions(self, integral_images, labels, correct_indices):
        """올바르게 예측된 TP, TN 개수를 세는 함수입니다."""
        true_positive, true_negative = 0, 0

        for idx in correct_indices:
            prediction = self.predict(integral_images[idx])
            actual_label = labels[idx]
            
            if prediction == 1 and actual_label == 1:
                true_positive += 1
            elif prediction == 0 and actual_label == 0:
                true_negative += 1

        return true_positive, true_negative

    def filter_true_negative(self, integral_images, labels):
        """True Negative를 제외하고 나머지 데이터를 필터링하여 반환합니다."""
        return [
            (img, label) for img, label in zip(integral_images, labels)
            if label != 0 or self.predict(img) != 0
        ]
    
    def get_incorrect_indices(self, integral_images, labels):
        incorrect_indices = []
        for i in range(len(labels)):
            prediction = self.predict(integral_images[i])
            actual_label = labels[i]
            
            # FN, FP만 잘못 예측한 경우에만 해당 인덱스 추가
            if (prediction == 1 and actual_label == 0) or (prediction == 0 and actual_label == 1):
                incorrect_indices.append(i)
                
        return incorrect_indices

    def process_predictions(self, stage, integral_images, labels):
        total_indices = range(len(labels))

        incorrect_indices = self.get_incorrect_indices(integral_images, labels)
        correct_indices = self.filter_correct_indices(total_indices, incorrect_indices)

        _, correct_tn = self.count_correct_predictions(
            integral_images, labels, correct_indices
        )
        logging.info(f"[Data] {stage}번째 데이터 {len(labels)}개 중, {correct_tn}개 제외")

        filtered_data = self.filter_true_negative(integral_images, labels)
        integral_images, labels = zip(*filtered_data)
        integral_images, labels = list(integral_images), list(labels)

        weights = np.ones(len(labels)) / len(labels)
        logging.info(f"-----------------------------------------------------------------------------------------------")
        
        return integral_images, labels, weights

    def train_weak_classifier_wrapper(self, args):
        return self.train_weak_classifier(*args)
    
    def train_weak_classifier(self, integral_images, labels, haar_like_filter, weights):
        weak_classifier = WeakClassifier(haar_like_filter)
        weak_classifier.train(integral_images, labels, weights)

        predictions = np.array([weak_classifier.predict(img) for img in integral_images])
        error = np.sum(weights * (predictions != labels))

        return weak_classifier, predictions, error
    
    def predict(self, integral_image):
        for stage_idx, (stage_weak_classifiers, stage_alphas) in enumerate(zip(self.weak_classifiers, self.alphas)):

            stage_prediction = 0
            for weak_classifier, alpha in zip(stage_weak_classifiers, stage_alphas):
                stage_prediction += alpha * weak_classifier.predict(integral_image)

            # 현재 단계에서 "얼굴 아님"이라면 바로 0을 반환
            if stage_prediction <= 0:
                # logging.info(f"Return 0 at Stage {stage_idx} - Stage Prediction: {stage_prediction}")
                return 0

        # 모든 단계에서 "얼굴 아님"이 아니면 최종적으로 1을 반환 (얼굴 있음)
        # logging.info(f"Return 1 after all stages - Final Stage Prediction: {stage_prediction}")
        return 1

    def evaluate(self, X, y):
        metrics = {}
        true_positive, true_negative = 0, 0
        false_positive, false_negative = 0, 0

        for i in range(len(y)):
            prediction = self.predict(X[i])
            if prediction == y[i]:
                if prediction == 1:
                    true_positive += 1
                else:
                    true_negative += 1
            else:
                if prediction == 1:
                    false_positive += 1
                else:
                    false_negative += 1

        metrics['true_positive'] = true_positive
        metrics['true_negative'] = true_negative
        metrics['false_positive'] = false_positive
        metrics['false_negative'] = false_negative
        
        # Calculate accuracy
        metrics['accuracy'] = (true_positive + true_negative) / (true_positive + false_negative + true_negative + false_positive)

        # Calculate precision
        if (true_positive + false_positive) > 0:
            metrics['precision'] = true_positive / (true_positive + false_positive)
        else:
            metrics['precision'] = 0

        # Calculate recall
        if (true_positive + false_negative) > 0:
            metrics['recall'] = true_positive / (true_positive + false_negative)
        else:
            metrics['recall'] = 0

        # Calculate F1 score if both precision and recall are greater than 0
        if metrics['precision'] > 0 and metrics['recall'] > 0:
            metrics['f1'] = (2 * metrics['precision'] * metrics['recall']) / (metrics['precision'] + metrics['recall'])
        else:
            metrics['f1'] = 0

        return metrics
    
    def save(self, file_path):
        with open(file_path, 'wb') as f:
            pickle.dump(self, f)
        logging.info(f'[Save] 모델이 {file_path}에 저장되었습니다.')
        logging.info(f"-----------------------------------------------------------------------------------------------")
    
    @classmethod
    def load(cls, file_path):
        with open(file_path, 'rb') as f:
            model = pickle.load(f)
        logging.info(f'[Loaded] 성공적으로 {file_path}로부터 모델을 불러왔습니다.')
        logging.info(f"-----------------------------------------------------------------------------------------------")
        
        return model
    