# coding: utf-8

import argparse
import os
import sys
import warnings

import numpy as np

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

# Root directory of the project
ROOT_DIR = os.path.abspath("../")
sys.path.append(ROOT_DIR)  # To find local version of the library

from mrcnn import utils
import mrcnn.model as modellib
import train


class InferenceConfig(train.objectConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1


if __name__ == '__main__':
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Predict Mask R-CNN to detect objects.')
    parser.add_argument('--dataset', required=True,
                        metavar="/path/to/object/dataset/",
                        help='Directory of the test object dataset')
    parser.add_argument('--weights', required=True,
                        metavar="/path/to/weights.h5",
                        help="Path to weights .h5 file or 'coco'")
    args = parser.parse_args()

    MODEL_DIR = os.path.join(ROOT_DIR, "logs")
    MODEL_PATH = os.path.join(ROOT_DIR, args.weights)
    IMAGE_DIR = args.dataset

    config = InferenceConfig()
    config.display()

    model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)
    model.load_weights(MODEL_PATH, by_name=True)

    dataset_test = train.objectDataset()
    dataset_test.load_object(IMAGE_DIR, "test")
    dataset_test.prepare()

    image_ids = np.random.choice(dataset_test.image_ids, 10)
    APs = []
    for image_id in image_ids:
        # Load image and ground truth data
        image, image_meta, gt_class_id, gt_bbox, gt_mask = \
            modellib.load_image_gt(dataset_test, config, image_id, use_mini_mask=False)
        molded_images = np.expand_dims(modellib.mold_image(image, config), 0)

        # Run object detection
        results = model.detect([image], verbose=0)
        r = results[0]

        # Compute AP
        AP, precisions, recalls, overlaps = \
            utils.compute_ap(gt_bbox, gt_class_id, gt_mask,
                             r["rois"], r["class_ids"], r["scores"], r['masks'])
        APs.append(AP)

    print("mAP: {}%".format(round(np.mean(APs) * 100, 4)))
