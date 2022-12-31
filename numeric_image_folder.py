# Author: Matt Williams
# Version: 12/26/2022

from torchvision.datasets.folder import default_loader
from torchvision.datasets import ImageFolder


class NumericImageFolder(ImageFolder):
    '''This Custom Image Folder gives each class an index
    depending on its numerical order instead of lexicographic order.
    This class assumes the dataset classes are labeled as non-negative integers.'''
    def __init__(self, root, transform = None, target_transform = None,
                loader = default_loader, is_valid_file = None):
        super(NumericImageFolder, self).__init__(root,
                                                transform=transform, 
                                                target_transform=target_transform,
                                                loader = loader,
                                                is_valid_file=is_valid_file) 

    def find_classes(self, directory): 
        classes, class_to_idx = super(NumericImageFolder, self).find_classes(directory)
        for cls_num in class_to_idx.keys():
            class_to_idx[cls_num] = int(cls_num)

        return classes, class_to_idx



    

