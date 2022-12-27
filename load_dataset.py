# Author: Matt Williams
# Version: 12/26/2022

from torchvision.datasets import MNIST, FashionMNIST, EMNIST
from torch.utils.data import DataLoader, Subset
from utils import MNIST_TRANSFORM, EMNIST_TRANSFORM, MNIST_BATCH_SIZE
from sklearn.model_selection import train_test_split

# File paths for downloading mnist related datasets
MNIST_TRAIN_PATH = '/tmp/mnist'
MNIST_TEST_PATH = '/tmp/mnist_test_'
FASH_MNIST_TRAIN_PATH = '/temp/fasion_mnist'
FASH_MNIST_TEST_PATH = '/temp/fasion_mnist_test_'
EMNIST_TRAIN_PATH = '/tmp/emnist'
EMNIST_TEST_PATH = '/tmp/emnist_test_'

RANDOM_SEED = 42
NUM_WORKERS = 4


def _make_train_valid_split(ds_train, len_ds_test):
    train_idxs, valid_idxs, _, _ = train_test_split(
            range(len(ds_train)),
            ds_train.targets,
            stratify=ds_train.targets,
            test_size= len(len_ds_test) / len(ds_train), 
            random_state=RANDOM_SEED
        )
    train_split = Subset(ds_train, train_idxs)
    valid_split = Subset(ds_train, valid_idxs)
    dl_train = DataLoader(train_split, batch_size=MNIST_BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    dl_valid = DataLoader(valid_split, batch_size=MNIST_BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    
    return dl_train, dl_valid

def _make_data_loaders(ds_train, ds_test, validate, return_test_ds = False):
    dl_test = DataLoader(ds_test, batch_size=MNIST_BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    dl_train = DataLoader(ds_train, batch_size=MNIST_BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    dl_valid = None
    if validate:
        dl_train, dl_valid = _make_train_valid_split(ds_train, len(ds_test))

    extra = ds_test if return_test_ds else None

    return dl_train, dl_valid, dl_test, extra


def load_mnist(validate = False, return_test_ds = False):
    ds_train = MNIST(MNIST_TRAIN_PATH, download=True, train=True, transform=MNIST_TRANSFORM)
    ds_test = MNIST(MNIST_TEST_PATH, download=True, train=False, transform=MNIST_TRANSFORM)
    return _make_data_loaders(ds_train, ds_test, validate, return_test_ds)


def load_fashion_mnist(validate=False, return_test_ds = False):
    ds_train = FashionMNIST(FASH_MNIST_TRAIN_PATH, download=True, train = True, transform=MNIST_TRANSFORM)
    ds_test = FashionMNIST(FASH_MNIST_TEST_PATH, download=True, train = False, transform=MNIST_TRANSFORM)
    return _make_data_loaders(ds_train, ds_test, validate, return_test_ds)


def load_emnist(split = "balanced", validate = False, return_test_ds = False):
    ds_train = EMNIST(EMNIST_TRAIN_PATH, split = split, download = True, train = True, transform = EMNIST_TRANSFORM)
    ds_test = EMNIST(EMNIST_TEST_PATH, split=split, download = True, train = False, transform = EMNIST_TRANSFORM)
    return _make_data_loaders(ds_train, ds_test, validate, return_test_ds)



