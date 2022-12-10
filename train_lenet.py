# Author: Matt Williams
# Version: 12/6/2022

from LeNet import LeNet_5
from torchvision import transforms
from torchvision.datasets import MNIST, FashionMNIST, EMNIST
from torch.utils.data import DataLoader, random_split
from utils import *
import matplotlib.pyplot as plt
import torch
import numpy as np
from sklearn.metrics import confusion_matrix
from torch.nn.modules import CrossEntropyLoss, MSELoss, AvgPool2d, MaxPool2d, Tanh, ReLU
from torch.optim import SGD, Adam

def save_training_metrics(model, dl_test, train_losses, valid_losses, save_visual_name, model_name, ds_name): 
    '''
    Args: 
    - model: Lenet Model instance.
    - dl_test: DataLoader containing test data.
    - train_losses: list of training loss values from training.
    - valid_losses: list of validation loss values from training. Could be empty.
    - save_visual_name: file name for the training and validation loss graph to be saved as.
    - model_name: name of the model. To be input into accuracies csv.
    - ds_name: name of the dataset. To be input into accuracies csv.
    '''
    plt.plot(range(1, len(train_losses) + 1), train_losses, label = "Training Loss")
    if len(valid_losses) > 0:
        plt.plot(range(1, len(valid_losses) + 1), valid_losses, label = "Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Losses")
    plt.title("Training Loss values")
    plt.legend()
    plt.savefig(os.path.join(VISUAL_DIR, save_visual_name))
    plt.clf()

def get_dataloaders(Data_Set_Type, train_download_path, test_download_path, validate, split = None): 
    '''
    Args:
    - Data_Set_Type: Dataset type reference (not instance) make data loaders for.
    - train_download_path: where to download the dataset's training information.
    - test_download_path: where to downoad the datset's testing information.
    - validate: should validation dataloader be made.
    '''
    ds_test = Data_Set_Type(test_download_path, download=True, train = False, transform=MNIST_TRANSFORM) \
                if split == None else \
                Data_Set_Type(test_download_path, download=True, train = False, split = split, transform=MNIST_TRANSFORM)

    dl_test = DataLoader(ds_test, batch_size=MNIST_BATCH_SIZE, shuffle = False, num_workers = 4)

    ds_train = Data_Set_Type(train_download_path, download=True, transform=MNIST_TRANSFORM) \
                if split == None else \
                Data_Set_Type(train_download_path, download=True, split = split, transform=MNIST_TRANSFORM)    

    ds_valid = None
    if validate:
        ds_train, ds_valid = random_split(ds_train, [len(ds_train) - len(ds_test),len(ds_test)])

    dl_train = DataLoader(ds_train, batch_size=MNIST_BATCH_SIZE, shuffle = True, num_workers=4)

    dl_valid = None
    if validate:
        dl_valid = DataLoader(ds_valid, batch_size=MNIST_BATCH_SIZE, shuffle = True, num_workers=4)

    ds_name = ds_train.__str__().split("\n")[0].split(" ")[1]

    return dl_train, dl_valid, dl_test, ds_name

def run_trad_lenet(dl_train, dl_valid, dl_test, save_path, save_visual_name, ds_name, num_classes,
                 early_stopping = False, validate = False, pad_first_conv = True):
    """
    Args: 
    - dl_train: data loader instance with our training information.
    - dl_valid: data loader instance with our validation information.
    - dl_test: data loader instance with our testing information.
    - save_path: file path to save the model to.
    - save_visual_name: file name for the training loss visual to be made.
    - ds_name: the name of the dataset.
    - early_stopping: should the model stop training early if validation loss increases.
    - validate: should the model perform validation after each epoch
    - pad_first_conv: boolean to determine if the first convolution layer should be padded or not. If
            the first convolution layer is padded, then an additional pooling layer is added.
    """
    model = LeNet_5(dl_train, 
                    dl_valid,
                    num_classes,
                    CrossEntropyLoss(),
                    AvgPool2d,
                    Tanh,
                    save_path,
                    pad_first_conv=pad_first_conv,
                    early_stopping=early_stopping
     )

    model.to(device)
    model.set_optimizer(SGD(model.parameters(), lr = 0.01, momentum = 0.9))
    train_losses, valid_losses = model.run_epochs(n_epochs=100, validate=validate)
    save_training_metrics(model, dl_test, train_losses, valid_losses, save_visual_name, "Trad. LeNet", ds_name)


def run_modern_lenet(dl_train, dl_valid, dl_test, save_path, save_visual_name, ds_name, num_classes,
                     early_stopping = False, validate = False, pad_first_conv = False):
    """
    Args: 
    - dl_train: data loader instance with our training information.
    - dl_valid: data loader instance with our validation information.
    - dl_test: data loader instance with our testing information.
    - save_path: file path to save the model to.
    - save_visual_name: file name for the training loss visual to be made.
    - ds_name: the name of the dataset.
    - early_stopping: should the model stop training early if validation loss increases.
    - validate: should the model perform validation after each epoch
    - pad_first_conv: boolean to determine if the first convolution layer should be padded or not. If
    """ 
     
    model = LeNet_5(dl_train,
                    dl_valid,
                    num_classes,
                    CrossEntropyLoss(),
                    MaxPool2d,
                    ReLU,
                    save_path,
                    pad_first_conv=pad_first_conv,
                    early_stopping=early_stopping
    )

    model = model.to(device)
    model.set_optimizer(Adam(model.parameters(), lr = 0.01))
    train_losses, valid_losses = model.run_epochs(n_epochs=100, validate=validate)
    save_training_metrics(model, dl_test, train_losses, valid_losses, save_visual_name, "Mod. LeNet", ds_name)

if __name__ == "__main__":

    # For traditional LeNet with regular mnist
    dl_train, dl_valid, dl_test, ds_name = get_dataloaders(MNIST, MNIST_TRAIN_PATH, MNIST_TEST_PATH, False)
    run_trad_lenet(dl_train, 
                    dl_valid, 
                    dl_test,
                    TRAD_LENET_MNIST_PATH, 
                    "Trad_LeNet_mnist.png", 
                    ds_name,
                    10
                    )
    
    # For traditional LeNet with fashion mnist
    dl_train, dl_valid, dl_test, ds_name = get_dataloaders(FashionMNIST, FASH_MNIST_TRAIN_PATH, FASH_MNIST_TEST_PATH, False)
    run_trad_lenet(dl_train,
                    dl_valid,
                    dl_test,
                    TRAD_LENET_FASH_MNIST_PATH,
                    "Trad_LeNet_fash_mnist.png", 
                    ds_name,
                     10)

    # For traditional LeNet with emnist
    dl_train, dl_valid, dl_test, ds_name = get_dataloaders(EMNIST, EMNIST_TRAIN_PATH, EMNIST_TEST_PATH, False, "balanced")
    run_trad_lenet(dl_train,
                    dl_valid,
                    dl_test,
                    TRAD_LENET_EMNIST_PATH,
                    "Trad_LeNet_emnist.png",
                    ds_name, 
                    47)

    # For Modern LeNet with regular mnist
    dl_train, dl_valid, dl_test, ds_name = get_dataloaders(MNIST, MNIST_TRAIN_PATH, MNIST_TEST_PATH, False)
    run_modern_lenet(dl_train,
                    dl_valid, dl_test,
                    MOD_LENET_MNIST_PATH,
                    "Modern_LeNet_mnist.png",
                    ds_name,
                    10)

    # For Modern LeNet with fashion mnist
    dl_train, dl_valid, dl_test, ds_name = get_dataloaders(FashionMNIST, FASH_MNIST_TRAIN_PATH, FASH_MNIST_TEST_PATH, False)
    run_modern_lenet(dl_train,
                    dl_valid,
                    dl_test,
                    MOD_LENET_FASH_MNIST_PATH,
                    "Modern_LeNet_fash_mnist.png",
                    ds_name,
                    10)

    # For Modern LeNet with emnist
    dl_train, dl_valid, dl_test, ds_name = get_dataloaders(EMNIST, EMNIST_TRAIN_PATH, EMNIST_TEST_PATH, False, "balanced")
    run_modern_lenet(dl_train,
                    dl_valid,
                    dl_test, 
                    MOD_LENET_EMNIST_PATH,
                    "Modern_LeNet_emnist.png",
                    ds_name,
                    47)

