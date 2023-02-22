# Author: Matt Williams
# Version: 1/26/2022

import numpy as np
from tqdm import tqdm
#import pytorch_lightning as pl
from matplotlib import pyplot as plt

# Signal related
'''
import lmdb
import pickle
from torchsig.datasets import ModulationsDataset
from torchsig.utils.dataset import SignalDataset
import torchsig.transforms as ST
from torchsig.utils.visualize import IQVisualizer, SpectrogramVisualizer
'''


from torchvision.datasets import MNIST, FashionMNIST, EMNIST
from torch.utils.data import DataLoader, Subset, ConcatDataset
from utils import *
from sklearn.model_selection import train_test_split
from slice_torch_dataset import TiledDataset
import os
from numeric_image_folder import NumericImageFolder

# File paths for downloading mnist related datasets
MNIST_TRAIN_PATH = '/tmp/mnist'
MNIST_TEST_PATH = '/tmp/mnist_test_'
FASH_MNIST_TRAIN_PATH = '/tmp/fasion_mnist'
FASH_MNIST_TEST_PATH = '/tmp/fasion_mnist_test_'
EMNIST_TRAIN_PATH = '/tmp/emnist'
EMNIST_TEST_PATH = '/tmp/emnist_test_'
SIG_TRAIN_PATH = '/tmp/sig'
SIG_TEST_PATH = '/tmp/sig_test_'


def _make_train_valid_split(dataset, len_ds_test):
    train_idxs, valid_idxs, _, _ = train_test_split(
            range(len(dataset)),
            dataset.targets,
            stratify = dataset.targets,
            test_size = len_ds_test / len(dataset), 
            random_state = RANDOM_SEED
        )
    ds_train = Subset(dataset, train_idxs)
    ds_valid = Subset(dataset, valid_idxs)
    
    return ds_train, ds_valid

def _make_data_loaders(ds_train, ds_test, validate, return_tiled, num_tiles, tile_split):
    """
    Arguments: 
    - ds_train : Training Dataset instance. 
    - ds_test : Testing Dataset instance."""

    ds_valid = None
    if validate:
        ds_train, ds_valid = _make_train_valid_split(ds_train, len(ds_test))

    if return_tiled:
        ds_train = TiledDataset(ds_train, num_tiles, tile_split)
        ds_test = TiledDataset(ds_test, num_tiles, tile_split)
        if ds_valid:
            ds_valid = TiledDataset(ds_valid, num_tiles, tile_split)

    dl_train = DataLoader(ds_train, batch_size=MNIST_BATCH_SIZE, shuffle=True, num_workers=NUM_DATA_LOADER_WORKERS)
    dl_test = DataLoader(ds_test, batch_size=MNIST_BATCH_SIZE, shuffle=False, num_workers=NUM_DATA_LOADER_WORKERS)
    dl_valid = None
    if ds_valid:
        dl_valid = DataLoader(ds_valid, batch_size=MNIST_BATCH_SIZE, shuffle=True, num_workers=NUM_DATA_LOADER_WORKERS)
    
    return dl_train, dl_valid, dl_test


def load_mnist(validate = False, return_tiled = False, num_tiles = 2, tile_split = "v"):
    ds_train = MNIST(MNIST_TRAIN_PATH, download=True, train=True, transform=MNIST_TRANSFORM)
    ds_test = MNIST(MNIST_TEST_PATH, download=True, train=False, transform=MNIST_TRANSFORM)
    return _make_data_loaders(ds_train, ds_test, validate, return_tiled, num_tiles, tile_split)


def load_fft_mnist(validate = False, return_tiled = False, num_tiles = 2, tile_split = "v"):
    ds_train = MNIST(MNIST_TRAIN_PATH, download=True, train = True, transform=FFT_MNIST_TRANSFORM)
    ds_test = MNIST(MNIST_TEST_PATH, download=True, train=False, transform=FFT_MNIST_TRANSFORM)
    return _make_data_loaders(ds_train, ds_test, validate, return_tiled, num_tiles, tile_split)


def load_fashion_mnist(validate=False, return_tiled = False, num_tiles = 2, tile_split = "v"):
    ds_train = FashionMNIST(FASH_MNIST_TRAIN_PATH, download=True, train = True, transform=MNIST_TRANSFORM)
    ds_test = FashionMNIST(FASH_MNIST_TEST_PATH, download=True, train = False, transform=MNIST_TRANSFORM)
    return _make_data_loaders(ds_train, ds_test, validate, return_tiled, num_tiles, tile_split)


def load_emnist(split = "balanced", validate = False, return_tiled = False, num_tiles = 2, tile_split = "v"):
    """
    Arguments: 
    - split: which EMNIST dataset split to download. 
    - validate: Should a validation dataset be made. It will be the same size as the test set.
    - return_tiled: Should the images in the dataset be split?
    - num_tiles: How many times should each image be split? Choose either 2 or 4.
    - tile_split: In what orientation should the image be split? Choose either h or v. If num_tiles = 4, 
    then this variable doesnt' matter."""
    ds_train = EMNIST(EMNIST_TRAIN_PATH, split = split, download = True, train = True, transform = EMNIST_TRANSFORM)
    ds_test = EMNIST(EMNIST_TEST_PATH, split = split, download = True, train = False, transform = EMNIST_TRANSFORM)
    return _make_data_loaders(ds_train, ds_test, validate, return_tiled, num_tiles, tile_split)


def load_mega_dataset(dataset_dir, transform = None, test_size = 0.2):
    training_subsets = []
    dl_test_map = {}

    for recon in os.listdir(dataset_dir):
        cur_dataset = NumericImageFolder(os.path.join(dataset_dir, recon), transform=transform)
        training_idxs, test_idxs, _, _ = train_test_split(range(len(cur_dataset)), 
                                                        cur_dataset.targets, 
                                                        stratify = cur_dataset.targets, 
                                                        test_size = test_size,
                                                        random_state=RANDOM_SEED)
        training_subsets.append(Subset(cur_dataset, training_idxs))
        dl_test_map[recon] = DataLoader(Subset(cur_dataset, test_idxs), batch_size=MNIST_BATCH_SIZE, shuffle=False, num_workers=NUM_DATA_LOADER_WORKERS)

    dl_concat = DataLoader(ConcatDataset(training_subsets), batch_size=MNIST_BATCH_SIZE, shuffle=True, num_workers=NUM_DATA_LOADER_WORKERS)
    return dl_concat, dl_test_map    


# Signal related
'''def load_sig(classes, level = 0, include_snr = False, dim_size = 32, samples_per_class = 100):
    classes = [
        "bpsk","8pam","8psk","16qam","16pam",
        "64qam","64psk","256qam","1024qam","16gmsk",
        ]

    level = 0

    include_snr = False

    num_classes = len(classes)
    num_iq_samples = dim_size * dim_size

    data_transform = ST.Compose([
        ST.Normalize(norm=np.inf), 
        ST.ComplexTo2D()
    ])

    
    # Reshaping does not work
    ST.Lambda(lambda x: SignalData(\
                data = x.iq_data.reshape((x.iq_data.shape[0], int(round(np.sqrt(x.iq_data.shape[1]))), int(round(np.sqrt(x.iq_data.shape[1])))))), \
                item_type=np.dtype(np.float64), 
                data_type=np.dtype(np.float64),
                signal_description=x.signal_description, \
            )

    # Seed the dataset instantiation for reporduceability
    pl.seed_everything(1234567891)

    ds_train = ModulationDataset(
        classes=classes,
        use_class_idx = True, #False,
        level = level, 
        num_iq_samples = 1024,
        num_samples = int(num_classes*samples_per_class),
        include_snr = include_snr, 
        transform = data_transform
    )

    ds_test_len = int(num_classes * samples_per_class/3)
    ds_train, ds_test = _make_train_valid_split(ds_train, ds_test_len)
    return _make_data_loaders(ds_train, ds_test, validate, return_tiled, num_tiles, tile_split)'''