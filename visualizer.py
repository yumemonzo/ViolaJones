import matplotlib.pyplot as plt


def plot_performance_metrics(stage_metrics, save_path):
    # 각 메트릭 리스트 초기화
    true_positive = []
    true_negative = []
    false_positive = []
    false_negative = []

    # stage_metrics에서 메트릭 추출
    for metrics in stage_metrics:
        true_positive.append(metrics['true_positive'])
        true_negative.append(metrics['true_negative'])
        false_positive.append(metrics['false_positive'])
        false_negative.append(metrics['false_negative'])

    # 그래프 그리기
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(stage_metrics)), true_positive, marker='o', label='True Positive', color='blue')
    plt.plot(range(len(stage_metrics)), true_negative, marker='o', label='True Negative', color='green')
    plt.plot(range(len(stage_metrics)), false_positive, marker='o', label='False Positive', color='red')
    plt.plot(range(len(stage_metrics)), false_negative, marker='o', label='False Negative', color='orange')

    plt.title('Stage-wise Performance Metrics')
    plt.xlabel('Stage')
    plt.ylabel('Count')
    plt.xticks(range(len(stage_metrics)))  # x축 눈금 설정
    plt.legend()
    plt.grid()

    # 지정된 경로에 이미지 저장
    plt.savefig(save_path)
    plt.close()


def plot_thresholds(weak_classifiers, save_path):
    num_stages = len(weak_classifiers)
    plt.figure(figsize=(10, 6))

    for stage_idx in range(num_stages):
        stage_thresholds = [wc.threshold for wc in weak_classifiers[stage_idx]]
        x_indices = [stage_idx + i / 10.0 for i in range(len(stage_thresholds))]  # 각 스테이지에 대해 인덱스 조정

        # Stage별 threshold를 플롯
        plt.plot(x_indices, stage_thresholds, marker='o', label=f'Stage {stage_idx} Threshold', linestyle='-')

    plt.title('Weak Classifier Thresholds by Stage')
    plt.xlabel('Weak Classifier Index (by Stage)')
    plt.ylabel('Threshold Value')
    plt.xticks([i for i in range(num_stages)])  # x축 눈금 설정
    plt.legend()
    plt.grid()

    # 지정된 경로에 이미지 저장
    plt.savefig(save_path)
    plt.close()  # 현재 플롯을 닫아 메모리 해제


def plot_alphas(alphas, save_path):
    num_stages = len(alphas)
    plt.figure(figsize=(10, 6))

    for stage_idx in range(num_stages):
        stage_alphas = alphas[stage_idx]
        x_indices = [stage_idx + i / 10.0 for i in range(len(stage_alphas))]  # 각 스테이지에 대해 인덱스 조정

        # Stage별 alpha를 플롯
        plt.plot(x_indices, stage_alphas, marker='x', label=f'Stage {stage_idx} Alpha', linestyle='--')

    plt.title('Weak Classifier Alphas by Stage')
    plt.xlabel('Weak Classifier Index (by Stage)')
    plt.ylabel('Alpha Value')
    plt.xticks([i for i in range(num_stages)])  # x축 눈금 설정
    plt.legend()
    plt.grid()

    # 지정된 경로에 이미지 저장
    plt.savefig(save_path)
    plt.close()  # 현재 플롯을 닫아 메모리 해제


def plot_label_length(label_length, save_path):
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(label_length)), label_length, marker='o', label='Train Labels Length', color='blue')

    plt.title('Train Labels Length per Stage')
    plt.xlabel('Stage')
    plt.ylabel('Number of Train Labels')
    plt.xticks(range(len(label_length)))  # x축 눈금 설정
    plt.legend()
    plt.grid()
    plt.savefig(save_path)
    plt.close()  # 그래프를 닫아 메모리에서 해제