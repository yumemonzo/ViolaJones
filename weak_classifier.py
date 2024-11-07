import numpy as np


class WeakClassifier:
    def __init__(self, haar_like_filter, threshold=None, polarity=None):
        self.haar_like_filter = haar_like_filter
        self.threshold = threshold
        self.polarity = polarity
    
    def train(self, integral_images, labels, weights):
        haar_like_feature_values = np.array([self.haar_like_filter.compute_score(img) for img in integral_images])
        possible_thresholds = np.linspace(np.min(haar_like_feature_values), np.max(haar_like_feature_values), num=10)

        best_error = float('inf')
        for threshold in possible_thresholds:
            for polarity in [1, -1]:
                predictions = np.where(polarity * haar_like_feature_values < polarity * threshold, 1, -1)
                error = np.sum(weights * (predictions != labels))
                
                if error < best_error:
                    best_error = error
                    self.threshold = threshold
                    self.polarity = polarity
    
    def predict(self, integral_image):
        haar_like_feature = self.haar_like_filter.compute_score(integral_image)
        if self.polarity * haar_like_feature < self.polarity * self.threshold:
            return 1
        else:
            return 0
        