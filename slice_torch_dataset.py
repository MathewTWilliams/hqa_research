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
        self._data, self._targets = self._tile_dataset(orig_dataset)

    def _tile_dataset(self, orig_dataset):
        data = []
        targets = []
        if self._num_tiles not in [2,4]:
            return
        if self._tile_split.lower() not in ["v", "h"]:
            return

        for image, target in orig_dataset:
            crops = self._crop_image(image)
            for crop in crops:
                data.append(crop)
                targets.append(target)

        targets = torch.LongTensor(targets)
        return data, targets
    
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


class CombinedDataSet(Dataset):
    def __init__(self, orig_dataset, num_tiles, tile_split= "v"):
        self._num_tiles = num_tiles
        self._tile_split = tile_split
        self._data, self._targets = self._combine_dataset(orig_dataset)
        
    def _combine_dataset(self, orig_dataset):
        data = []
        targets = []

        if self._num_tiles not in [2,4]:
            return
        if self._tile_split.lower() not in ["v", "h"]:
            return

        for i in range(0, len(orig_dataset), self._num_tiles):
            target = orig_dataset[i][1]
            img_1 = orig_dataset[i][0].squeeze(0)
            img_2 = orig_dataset[i+1][0].squeeze(0)
            if self._num_tiles == 2 and self._tile_split == "v":
                combined_img = torch.concat([img_1, img_2], dim = 1)
                data.append(combined_img.unsqueeze(0))
            elif self._num_tiles == 2 and self._tile_split == "h":
                combined_img = torch.concat([img_1, img_2], dim = 0)
                data.append(combined_img.unsqueeze(0))
            elif self._num_tiles == 4:
                img_3 = orig_dataset[i+2][0].squeeze(0)
                img_4 = orig_dataset[i+3][0].squeeze(0)
                combined_img_top = torch.concat([img_1, img_2], dim = 1)
                combined_img_bot = torch.concat([img_3, img_4], dim = 1)
                combined_img = torch.concat([combined_img_top, combined_img_bot], dim = 0)
                data.append(combined_img.unsqueeze(0))

            targets.append(target)

        targets = torch.LongTensor(targets)
        return data, targets


    def __getitem__(self, index):
        return self._data[index], int(self._targets[index])

    def __len__(self):
        return len(self._data)
