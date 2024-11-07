import os
import logging
import argparse
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from haar_like_feature import build_haar_like_filters
from adaboost import AdaBoost
from utils import get_korea_time, create_directory_with_timestamp, load_images_to_array, compute_integral_image, process_image


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--layers', type=str,required=True)
    parser.add_argument('--train', action='store_true', help="Train mode enabled")
    parser.add_argument('--vis', action='store_true', help="Visualization enabled")
    parser.add_argument('--train_face_images_dir', type=str, default="/workspace/ViolaJones/data/train/faces")
    parser.add_argument('--train_background_images_dir', type=str, default="/workspace/ViolaJones/data/train/backgrounds")
    parser.add_argument('--test_face_images_dir', type=str, default="/workspace/ViolaJones/data/test/faces")
    parser.add_argument('--test_background_images_dir', type=str, default="/workspace/ViolaJones/data/test/backgrounds")
    parser.add_argument('--save_dir', type=str, default="/workspace/ViolaJones/result")
    parser.add_argument('--adaboost_path', type=str)

    return parser.parse_args()


def main():
    #######################################################################################################
    args = get_args()
    #######################################################################################################
    save_dir = create_directory_with_timestamp(args.save_dir, args.layers)
    #######################################################################################################
    logging.basicConfig(filename=os.path.join(save_dir, 'output_log.txt'), level=logging.INFO, format='%(message)s')
    start_time = get_korea_time()
    logging.info(f"-----------------------------------------------------------------------------------------------")
    logging.info(f"Start time: {start_time}")
    logging.info(f"-----------------------------------------------------------------------------------------------")
    #######################################################################################################
    train_face_images = load_images_to_array(args.train_face_images_dir)[:50]
    train_background_images = load_images_to_array(args.train_background_images_dir)[:50]
    logging.info(f"[Loaded] train_face_images length: {len(train_face_images)}")
    logging.info(f"[Loaded] train_background_images length: {len(train_background_images)}")
    logging.info(f"-----------------------------------------------------------------------------------------------")
    #######################################################################################################
    train_X = np.concatenate([train_face_images, train_background_images], axis=0)
    logging.info(f"[Generated] train_X length: {len(train_X)}")

    train_y = np.zeros(train_X.shape[0])
    train_y[:len(train_face_images)] = 1
    logging.info(f"[Generated] train_y length: {len(train_y)}")

    train_integral_images = []
    for X in train_X:
        train_integral_images.append(compute_integral_image(X))
    logging.info(f"[Generated] train_integral_images length: {len(train_integral_images)}")
    logging.info(f"-----------------------------------------------------------------------------------------------")
    #######################################################################################################
    test_face_images = load_images_to_array(args.test_face_images_dir)[:50]
    test_background_images = load_images_to_array(args.test_background_images_dir)[:50]
    logging.info(f"[Loaded] test_face_images length: {len(test_face_images)}")
    logging.info(f"[Loaded] test_background_images length: {len(test_background_images)}")
    logging.info(f"-----------------------------------------------------------------------------------------------")
    #######################################################################################################
    test_X = np.concatenate([test_face_images, test_background_images], axis=0)
    logging.info(f"[Generated] test_X length: {len(test_X)}")

    test_y = np.zeros(test_X.shape[0])
    test_y[:len(test_face_images)] = 1
    logging.info(f"[Generated] test_y length: {len(test_y)}")

    test_integral_images = []
    for X in test_X:
        test_integral_images.append(compute_integral_image(X))
    logging.info(f"[Generated] test_integral_images length: {len(test_integral_images)}")
    logging.info(f"-----------------------------------------------------------------------------------------------")
    #######################################################################################################
    haar_like_filters = build_haar_like_filters(train_X.shape[1], train_X.shape[2])
    logging.info(f"[Generated] haar_like_filters length: {len(haar_like_filters)}")
    logging.info(f"-----------------------------------------------------------------------------------------------")
    #######################################################################################################
    layers = list(map(int, args.layers.split(',')))
    
    adaboost = AdaBoost(layers)
    if args.train:
        adaboost.train(train_integral_images, train_y, test_integral_images, test_y, haar_like_filters, save_dir)
        adaboost.save(os.path.join(save_dir, f"layer_{'_'.join(map(str, layers))}_{len(train_y)}_{len(test_y)}"))
    else:
        adaboost.load(args.adaboost_path)
    #######################################################################################################
    if args.vis:
        file_names = os.listdir("/workspace/data/celebA/test_faces")
        logging.info(f"[Generated] Start visualization")
        with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
            futures = [executor.submit(process_image, file_name, "/workspace/data/celebA/test_faces", save_dir, adaboost) for file_name in file_names]
            for future in futures:
                future.result()
        logging.info(f"[Generated] Finish visualization")
        logging.info(f"-----------------------------------------------------------------------------------------------")
    #######################################################################################################

if __name__ == "__main__":
    main()
