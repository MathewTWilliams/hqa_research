# Author: Matt Williams
# Version: 12/6/2022

from lenet import Lenet_5
from utils import *
import matplotlib.pyplot as plt
from load_datasets import load_mnist, load_fashion_mnist, load_emnist
import os

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

    if not os.path.isdir(MISC_VIS_DIR):
        os.makedirs(MISC_VIS_DIR)

    plt.plot(range(1, len(train_losses) + 1), train_losses, label = "Training Loss")
    if len(valid_losses) > 0:
        plt.plot(range(1, len(valid_losses) + 1), valid_losses, label = "Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Losses")
    plt.title("Training Loss values")
    plt.legend()
    plt.savefig(os.path.join(MISC_VIS_DIR, save_visual_name))
    plt.clf()

def run_lenet(dl_train, dl_valid, dl_test, save_path, save_visual_name, ds_name, num_classes,
                 stop_early = False, validate = False):
    """
    Args: 
    - dl_train: data loader instance with our training information.
    - dl_valid: data loader instance with our validation information.
    - dl_test: data loader instance with our testing information.
    - save_path: file path to save the model to.
    - save_visual_name: file name for the training loss visual to be made.
    - ds_name: the name of the dataset.
    - num_classes: how many classes are in the dataset
    - stop_early: should the model stop training early if validation loss increases.
    - validate: should the model perform validation after each epoch
    """
    model = Lenet_5(dl_train, dl_valid, num_classes, save_path, stop_early)
    train_losses, valid_losses = model.run_epochs(n_epochs= 50, validate=validate)
    save_training_metrics(model, dl_test, train_losses, valid_losses, save_visual_name, "Lenet", ds_name)

if __name__ == "__main__":

    # For traditional LeNet with regular mnist
    dl_train, dl_valid, dl_test = load_mnist(validate=True)
    run_lenet(dl_train, 
            dl_valid, 
            dl_test,
            LENET_MNIST_PATH, 
            "Lenet_mnist.png", 
            "MNIST",
            10)
    
    # For traditional LeNet with fashion mnist
    dl_train, dl_valid, dl_test = load_fashion_mnist(validate=True)
    run_lenet(dl_train,
            dl_valid,
            dl_test,
            LENET_FASH_MNIST_PATH,
            "Lenet_fash_mnist.png", 
            "Fashion_MNIST",
             10)

    # For traditional LeNet with emnist
    dl_train, dl_valid, dl_test = load_emnist(validate=True)
    run_lenet(dl_train,
            dl_valid,
            dl_test,
            LENET_EMNIST_PATH,
            "Lenet_emnist.png",
            "EMNIST", 
            47)

