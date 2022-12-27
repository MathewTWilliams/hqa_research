# Author: Matt Williams
# Version: 12/26/2022

import torch
from torch.utils.data import Dataset
from numeric_image_folder import NumericImageFolder
from utils import IMG_MNIST_DIR_PATH
import os
from PIL import Image

class TiledDataset(Dataset):
    '''
    Usage:
    - Use an ImageFolder class to load in data like we are training Lenet.
    - 
    '''
    def __init__(self, images_targets, transform = None, target_transform = None,
                 num_tiles = 2, tile_split = "v"):

        self._transform = transform
        self._target_transform = target_transform
        self._num_tiles = num_tiles
        self._tile_split = tile_split
        self.tile_dataset(images_targets)

    def tile_dataset(self, images_targets):
        if self._num_tiles not in [2,4]:
            return
        if self._tile_split.lower() not in ["v", "h"]:
            return

        self._data = []
        self._targets = []
        for image_path, target in images_targets:
            image = Image.open(image_path)
            results = self.crop_image(image)
            results[0].show()
            results[1].show()
            results[2].show()
            results[3].show()
            return
    
    def crop_image(self, img):
        crops = []
        half_width = img.width // 2
        half_height = img.height // 2

        if self._num_tiles == 2 and self._tile_split == "v":
            box_l= (0, 0, half_width, img.height)
            box_r = (half_width+1, 0, img.width, img.height)
            crops.append(img.crop(box_l))
            crops.append(img.crop(box_r))

        elif self._num_tiles == 2 and self._tile_split == "h":
            box_t = (0, 0, img.width, half_height)
            box_b = (0, half_height+1, img.width, img.height)
            crops.append(img.crop(box_t))
            crops.append(img.crop(box_b))

        elif self._num_tiles == 4:
            box_tl = (0, 0, half_width, half_height)
            box_tr= (half_width + 1, 0, img.width, half_height)
            box_bl = (0, half_height+ 1, half_width, img.height)
            box_br = (half_width+1, half_height+1, img.width, img.height)
            crops.append(img.crop(box_tl))
            crops.append(img.crop(box_tr))
            crops.append(img.crop(box_bl))
            crops.append(img.crop(box_br))

        return crops

if __name__ == "__main__":
    root = "data_original"
    image_ds = NumericImageFolder(os.path.join(IMG_MNIST_DIR_PATH, root))
    tiled_ds = TiledDataset(images_targets = image_ds.imgs, 
                            transform = image_ds.transform,
                            target_transform=image_ds.target_transform,
                            num_tiles=4)