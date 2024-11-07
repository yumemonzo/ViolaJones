import os
import pytz
import cv2
import numpy as np
from datetime import datetime
from PIL import Image


def calculate_bbox_overlap(bbox_A, bbox_B):
    xA_min, yA_min, xA_max, yA_max = bbox_A
    xB_min, yB_min, xB_max, yB_max = bbox_B

    inter_x_min = max(xA_min, xB_min)
    inter_y_min = max(yA_min, yB_min)
    inter_x_max = min(xA_max, xB_max)
    inter_y_max = min(yA_max, yB_max)

    inter_width = max(0, inter_x_max - inter_x_min)
    inter_height = max(0, inter_y_max - inter_y_min)
    inter_area = inter_width * inter_height

    bbox_B_area = (xB_max - xB_min) * (yB_max - yB_min)
    overlap_ratio = inter_area / bbox_B_area if bbox_B_area > 0 else 0
    
    return overlap_ratio


def get_korea_time():
    korea_tz = pytz.timezone('Asia/Seoul')
    korea_time = datetime.now(korea_tz)

    return korea_time.strftime('%Y-%m-%d %H:%M:%S')


def create_directory_with_timestamp(save_dir, layers):
    current_time = get_korea_time()
    timestamp = current_time.replace(":", "_").replace(" ", "_")

    folder_name = f"{timestamp}____{layers.replace(',', '_')}"
    folder_path = os.path.join(save_dir, folder_name)

    os.makedirs(folder_path, exist_ok=True)

    return folder_path


def load_images_to_array(folder_path):
    images_array = []
    for filename in os.listdir(folder_path):
        img_path = os.path.join(folder_path, filename)

        if os.path.isfile(img_path):
            images_array.append(np.array(Image.open(img_path).convert("L")))
    
    return np.array(images_array)


def compute_integral_image(image):
    height, width = image.shape
    sum_array = np.zeros(image.shape, dtype=np.uint32)
    integral_image = np.zeros(image.shape, dtype=np.uint32)

    for w in range(width):
        for h in range(height):
            above_sum = sum_array[h - 1, w] if h > 0 else 0
            left_integral = integral_image[h, w - 1] if w > 0 else 0
            current_value = image[h, w]

            sum_array[h, w] = above_sum + current_value
            integral_image[h, w] = left_integral + sum_array[h, w]

    return integral_image


def draw_bbox_on_image_with_multiscale_sliding_window(image, clf, min_window_size=(19, 19), max_window_size=(100, 100), scale_factor=1.25, step_size=5, threshold=0):
    """
    다양한 크기의 슬라이딩 윈도우를 통해 이미지에 얼굴 경계 상자를 그린 후, 
    모든 겹치는 상자를 포함하는 가장 큰 상자를 하나 그리는 함수.
    
    Parameters:
        image (numpy.ndarray): 컬러 원본 이미지.
        clf (AdaBoost): 학습된 AdaBoost 모델.
        min_window_size (tuple): 윈도우의 최소 크기 (너비, 높이).
        max_window_size (tuple): 윈도우의 최대 크기 (너비, 높이).
        scale_factor (float): 윈도우 크기를 증가시키는 비율.
        step_size (int): 슬라이딩 윈도우 이동 크기.
        threshold (float): 얼굴로 인식하기 위한 최소 stage prediction 값.

    Returns:
        numpy.ndarray: 얼굴이 감지된 경우 가장 큰 경계 상자가 그려진 컬러 이미지.
    """

    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    height, width = gray_image.shape
    min_win_w, min_win_h = min_window_size
    max_win_w, max_win_h = max_window_size

    # 감지된 경계 상자와 점수를 저장할 리스트
    bboxes = []
    scores = []

    # 윈도우 크기를 점진적으로 증가시키며 슬라이딩 윈도우 적용
    win_w, win_h = min_win_w, min_win_h
    while win_w <= max_win_w and win_h <= max_win_h:
        for y in range(0, height - win_h + 1, step_size):
            for x in range(0, width - win_w + 1, step_size):
                sub_image = gray_image[y:y + win_h, x:x + win_h]
                integral_sub_image = compute_integral_image(sub_image)

                # 얼굴 감지 예측 및 신뢰도 점수
                score = clf.predict(integral_sub_image)
                if score > threshold:
                    bboxes.append((x, y, x + win_w, y + win_h))
                    scores.append(score)

        # 윈도우 크기 확대
        win_w = int(win_w * scale_factor)
        win_h = int(win_h * scale_factor)

    # 모든 겹치는 상자를 포함하는 가장 큰 상자 계산 및 그리기
    if bboxes:
        for box in bboxes:
            cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), (0, 0, 255), 2)

    return image


def process_image(file_name, images_dir, save_dir, clf):
    """
    단일 이미지를 처리하여 얼굴 검출을 수행하고 저장합니다.

    Parameters:
        file_name (str): 이미지 파일 이름.
        images_dir (str): 원본 이미지가 저장된 디렉토리 경로.
        save_dir (str): 처리된 이미지를 저장할 디렉토리 경로.
        clf (AdaBoost): 학습된 AdaBoost 모델.
    """
    
    image = cv2.imread(os.path.join(images_dir, file_name))
    annotated_image = draw_bbox_on_image_with_multiscale_sliding_window(image, clf)
    cv2.imwrite(os.path.join(save_dir, file_name), annotated_image)