# Author: Matt Williams
# Version: 11/13/2022


import numpy as np
from utils import IMG_DIR_PATH, add_compression_rate

import os

def get_sizes(root):
    sizes = []

    for sub_dir in os.listdir(root):
        cur_sub_path = os.path.join(root, sub_dir)
        for img_file in os.listdir(cur_sub_path):
            cur_img_path = os.path.join(cur_sub_path, img_file)
            sizes.append(os.path.getsize(cur_img_path))

    return sizes


def main(): 

    orig_sizes = get_sizes(os.path.join(IMG_DIR_PATH, "data_original"))
    jpg_sizes = get_sizes(os.path.join(IMG_DIR_PATH, "data_jpg"))

    compression_rate = np.sum(np.divide(jpg_sizes, orig_sizes)) / len(orig_sizes)
    add_compression_rate("data_jpg", compression_rate)


if __name__ == "__main__": 
    main()
