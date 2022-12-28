# Author: Matt Williams
# Version: 12/26/2022

import torch
from torch.utils.data import Dataset
from torchvision.transforms.functional import crop

class TiledDataset(Dataset):
    '''
    Usage:
    - Use an ImageFolder class to load in data like we are training Lenet.
    - 
    '''
    def __init__(self, orig_dataset, num_tiles = 2, tile_split = "v"):

        self._num_tiles = num_tiles
        self._tile_split = tile_split
        self._tile_dataset(orig_dataset)

    def _tile_dataset(self, orig_dataset):
        self._data = []
        self._targets = []
        if self._num_tiles not in [2,4]:
            return
        if self._tile_split.lower() not in ["v", "h"]:
            return

        for image, target in orig_dataset:
            crops = self._crop_image(image)
            for crop in crops:
                self._data.append(crop)
                self._targets.append(target)
    
    def _crop_image(self, img):
        crops = []
        img_width = img.shape[1]
        img_height = img.shape[2]
        half_width = img_width // 2
        half_height = img_height // 2

        if self._num_tiles == 2 and self._tile_split == "v":
            crops.append(crop(img, 0, 0, img_height, half_width))
            crops.append(crop(img, 0, half_width+1, img_height, half_width))

        elif self._num_tiles == 2 and self._tile_split == "h":
            crops.append(crop(img, 0, 0, half_height, img_width))
            crops.append(crop(img, half_height+1, 0, half_height, img_width))

        elif self._num_tiles == 4:
            crops.append(crop(img, 0, 0, half_width, half_height))
            crops.append(crop(img, 0, half_width+1, half_height, half_width))
            crops.append(crop(img, half_height+1, 0, half_height, half_width))
            crops.append(crop(img, half_height+1, half_width+1, half_height, half_width))

        return crops


    def __getitem__(self, index):
        return self._data[index], self._targets[index]
    
    def __len__(self):
        return len(self._data)



    