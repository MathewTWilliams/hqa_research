# Author: Matt Williams
# Version: 12/6/2022

from lenet import Lenet_5
from utils import *
import matplotlib.pyplot as plt
from load_datasets import load_mnist, load_fashion_mnist, load_emnist, load_mega_dataset
import os
from pytorch_cnn_base import query_model
from evaluate_recons import evaluate_dataset
from torchattacks import FGSM

def save_training_metrics(train_losses, valid_losses, save_visual_name): 
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

def run_lenet(dl_train, dl_valid, save_path, save_visual_name, num_classes, stop_early = False, validate = False, attack = None):
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
    train_losses, valid_losses = model.run_epochs(n_epochs= 50, validate=validate, attack=attack)
    save_training_metrics(train_losses, valid_losses, save_visual_name)
    return model

def test_recon_model(model, dl_test_map, model_name, ds_name, attack = "None"): 
    model.eval()
    for recon, dl_test in dl_test_map.items(): 
        predictions = query_model(model, dl_test)
        labels = np.asarray([dl_test.dataset.dataset.targets[i] for i in dl_test.dataset.indices])
        _ = evaluate_dataset(model_name, labels, predictions, ds_name, recon, attack = attack)

if __name__ == "__main__":

    # Regular MNIST
    dl_train, dl_valid, _ = load_mnist(validate=True)
    lenet_model = run_lenet(dl_train, 
            dl_valid, 
            LENET_MNIST_PATH, 
            "Lenet_mnist.png", 
            "MNIST",
            10,
            validate = True)
    
    # Adversarial training on MNIST
    fgsm_attack = FGSM(lenet_model)
    _ = run_lenet(dl_train, 
            None, 
            LENET_ADV_MNIST_PATH, 
            "Lenet_adv_mnist.png",
            "Adv. MNIST", 
            10, 
            attack = fgsm_attack,
            validate = True)
    
    del lenet_model

    # Fashion MNIST
    dl_train, dl_valid, _ = load_fashion_mnist(validate=True)
    _ = run_lenet(dl_train,
            dl_valid,
            LENET_FASH_MNIST_PATH,
            "Lenet_fash_mnist.png", 
            "Fashion_MNIST",
             10,
             validate = True)

    # EMNIST
    dl_train, dl_valid, _ = load_emnist(validate=True)
    _ = run_lenet(dl_train,
            dl_valid,
            LENET_EMNIST_PATH,
            "Lenet_emnist.png",
            "EMNIST", 
            47, 
            validate = True)

    # Lenet training on mega dataset or reconstructions
    dl_train, dl_test_map = load_mega_dataset(IMG_MNIST_DIR_PATH, MNIST_TRANSFORM)
    lenet_model = run_lenet(
        dl_train,
        None,
        LENET_MNIST_RECONS_PATH, 
        "Lenet_mnist_recons.png",
        "MNIST Recons",
        10
    )

    # get the accuracies from lenet trainied on reconstructions
    test_recon_model(lenet_model, dl_test_map, "Lenet", "MNIST Recons")

