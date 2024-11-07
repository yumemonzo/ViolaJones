import os
from PIL import Image
from torchvision import datasets, transforms
from utils import calculate_bbox_overlap


def generate_data(dataset, face_dir, non_face_dir, max_count=2000, overlap_threshold=0.4, padding=20, crop_size=20, step_size=10):
    os.makedirs(face_dir, exist_ok=True)
    os.makedirs(non_face_dir, exist_ok=True)

    face_count, background_count = 0, 0
    for num in range(len(dataset)):
        """
        get gt_bbox
        """
        image, landmarks = dataset[num]
        landmarks = landmarks.numpy()

        landmark_coords = [
            (landmarks[0], landmarks[1]),  # left eye
            (landmarks[2], landmarks[3]),  # right eye
            (landmarks[4], landmarks[5]),  # nose
            (landmarks[6], landmarks[7]),  # left mouth
            (landmarks[8], landmarks[9])   # right mouth
        ]

        x_coords = [coord[0] for coord in landmark_coords]
        y_coords = [coord[1] for coord in landmark_coords]
        min_x, max_x = min(x_coords), max(x_coords)
        min_y, max_y = min(y_coords), max(y_coords)

        gt_bbox = [
            max(0, min_x - padding),
            max(0, min_y - padding),
            min(image.shape[2], max_x + padding),
            min(image.shape[1], max_y + padding)
        ]
        """
        get crop_bbox & generate data
        """
        image_numpy = (image.permute(1, 2, 0) * 255).byte().numpy()
        image_pil = Image.fromarray(image_numpy)
        for y in range(0, image_numpy.shape[0] - crop_size, step_size):
            for x in range(0, image_numpy.shape[1] - crop_size, step_size):
                cropped_box = [x, y, x + crop_size, y + crop_size]
                cropped_img = image_pil.crop((x, y, x + crop_size, y + crop_size))
                overlap = calculate_bbox_overlap(gt_bbox, cropped_box)

                if overlap >= overlap_threshold:
                    if face_count < max_count:
                        cropped_img.save(os.path.join(face_dir, f'{face_count}.jpg'))
                        face_count += 1
                else:
                    if background_count < max_count:
                        cropped_img.save(os.path.join(non_face_dir, f'{background_count}.jpg'))
                        background_count += 1

        if face_count >= max_count and background_count >= max_count:
            break
    print(f"데이터 생성을 위해 사용된 이미지 개수: {num+1}")
    print(f"생성된 얼굴 데이터 개수: {face_count}")
    print(f"생성된 배경 데이터 개수: {background_count}")


def main():
    data_transform = transforms.Compose([
        transforms.ToTensor()
    ])
    train_dataset = datasets.CelebA(root='./data', split='train', target_type=['landmarks'], transform=data_transform, download=True)
    test_dataset = datasets.CelebA(root='./data', split='test', target_type=['landmarks'], transform=data_transform, download=True)

    print(f"Train Dataset")
    generate_data(train_dataset, "/workspace/ViolaJones/data/train/faces", "/workspace/ViolaJones/data/train/backgrounds")
    print(f"Test Dataset")
    generate_data(test_dataset, "/workspace/ViolaJones/data/test/faces", "/workspace/ViolaJones/data/test/backgrounds")

if __name__ == "__main__":
    main()
