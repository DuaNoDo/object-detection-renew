# coding: utf-8

import argparse
import os
import sys
import warnings
from os import listdir
from os.path import isfile, join

import skimage.io

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

ROOT_DIR = os.path.abspath("../")
sys.path.append(ROOT_DIR)
sys.path.append(os.path.join(ROOT_DIR, "code/"))

import mrcnn.model as modellib
from mrcnn import visualize
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
    IMAGE_DIR = os.path.join(args.dataset, "test")

    config = InferenceConfig()
    config.display()

    model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)
    model.load_weights(MODEL_PATH, by_name=True)

    class_names = ['BG', 'object']
    imgs = [join(IMAGE_DIR, f) for f in listdir(IMAGE_DIR) if
            isfile(join(IMAGE_DIR, f)) and f.lower().endswith(('.png', '.jpg'))]
    for path in imgs:
        image = skimage.io.imread(path)
        if image.shape[-1] == 4:
            image = image[..., :3]

        # Run detection
        results = model.detect([image], verbose=1)

        # Visualize & save results
        r = results[0]
        count = r.get("rois").shape[0]
        basename = os.path.basename(path)
        save_name = "{}_detect_count{}.{}".format(basename.split('.')[0], count, basename.split('.')[1])

        visualize.save_result_box(image, r['rois'], r['masks'], r['class_ids'], class_names, IMAGE_DIR,
                                  save_name, r['scores'], auto_show=False)

    print("finish")
