# Author: Matt Williams
# Version: 12/26/2022

from torchvision.datasets import MNIST, FashionMNIST, EMNIST
from torch.utils.data import DataLoader, Subset
from utils import MNIST_TRANSFORM, EMNIST_TRANSFORM, MNIST_BATCH_SIZE, NUM_DATA_LOADER_WORKERS, \
RANDOM_SEED, FFT_MNIST_TRANSFORM
from sklearn.model_selection import train_test_split
from slice_torch_dataset import TiledDataset

# File paths for downloading mnist related datasets
MNIST_TRAIN_PATH = '/tmp/mnist'
MNIST_TEST_PATH = '/tmp/mnist_test_'
FASH_MNIST_TRAIN_PATH = '/tmp/fasion_mnist'
FASH_MNIST_TEST_PATH = '/tmp/fasion_mnist_test_'
EMNIST_TRAIN_PATH = '/tmp/emnist'
EMNIST_TEST_PATH = '/tmp/emnist_test_'

def _make_train_valid_split(ds_train, len_ds_test):
    train_idxs, valid_idxs, _, _ = train_test_split(
            range(len(ds_train)),
            ds_train.targets,
            stratify=ds_train.targets,
            test_size= len_ds_test / len(ds_train), 
            random_state=RANDOM_SEED
        )
    ds_train = Subset(ds_train, train_idxs)
    ds_valid = Subset(ds_train, valid_idxs)
    
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



