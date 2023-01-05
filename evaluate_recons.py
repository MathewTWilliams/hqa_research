# Author: Matt Williams
# Version: 12/26/2022

import pandas as pd
import numpy as np
from utils import *
from torch.utils.data import DataLoader
from torchvision import transforms
import os
import torch
from sklearn.metrics import confusion_matrix
import torchattacks
from tqdm import tqdm
from numeric_image_folder import NumericImageFolder
from slice_torch_dataset import CombinedDataSet
from pytorch_cnn_base import run_predictions

def evaluate_dataset(model_name, test_labels, predictions, ds_name, recon_name, attack = None):
    diag_accuracies = confusion_matrix(test_labels, predictions, normalize="true").diagonal().tolist()
    avg_accuracy = np.sum(diag_accuracies) / len(diag_accuracies)
    attack_name = attack if attack != None else "None"
    add_accuracy_results(model_name, ds_name, recon_name, attack_name, avg_accuracy)


def eval_model(model_save_path, model_name, dataset):

    lenet_model = torch.load(model_save_path)
    lenet_model.eval()

    ds_name = dataset.split("\\")[-1]

    for root in tqdm(RECON_ROOT_NAMES):
        ds_test = NumericImageFolder(os.path.join(dataset, root), transform=IMG_FOLDER_TRANFORM)
        dl_test = DataLoader(ds_test, batch_size=MNIST_BATCH_SIZE, shuffle = False, num_workers = NUM_DATA_LOADER_WORKERS)
        predictions = run_predictions(lenet_model, dl_test)
        evaluate_dataset(model_name, ds_test.targets, predictions, ds_name, root)

    fgsm_attack = torchattacks.FGSM(lenet_model)

    for root in tqdm(RECON_ROOT_NAMES):
        ds_test = NumericImageFolder(os.path.join(dataset, root), transform=IMG_FOLDER_TRANFORM)
        dl_test = DataLoader(ds_test, batch_size=MNIST_BATCH_SIZE, shuffle = False, num_workers = NUM_DATA_LOADER_WORKERS)
        predictions = run_predictions(lenet_model, dl_test, fgsm_attack)
        evaluate_dataset(model_name, ds_test.targets, predictions, ds_name, root, fgsm_attack.attack)


def eval_tiled_model(model_save_path, model_name, dataset):

    lenet_model = torch.load(model_save_path)
    lenet_model.eval()

    ds_name = dataset.split("\\")[-1]

    for i in tqdm(range(len(RECON_ROOT_NAMES) - 1)):
        root = RECON_ROOT_NAMES[i]
        ds_test = NumericImageFolder(os.path.join(dataset, root), transform=IMG_FOLDER_TRANFORM)
        ds_test = CombinedDataSet(ds_test, num_tiles=2, tile_split="v")
        dl_test = DataLoader(ds_test, batch_size=MNIST_BATCH_SIZE, shuffle = False, num_workers = NUM_DATA_LOADER_WORKERS)
        evaluate_dataset(lenet_model, model_name, dl_test, ds_name, root)

    fgsm_attack = torchattacks.FGSM(lenet_model)

    for i in tqdm(range(len(RECON_ROOT_NAMES) - 1)):
        root = RECON_ROOT_NAMES[i]
        ds_test = NumericImageFolder(os.path.join(dataset, root), transform=IMG_FOLDER_TRANFORM)
        ds_test = CombinedDataSet(ds_test, num_tiles=2, tile_split="v")
        dl_test = DataLoader(ds_test, batch_size=MNIST_BATCH_SIZE, shuffle = False, num_workers = NUM_DATA_LOADER_WORKERS)
        evaluate_dataset(lenet_model, model_name, dl_test, ds_name, root, fgsm_attack)
    
def main():

    #eval_model(LENET_MNIST_PATH, "Lenet", IMG_MNIST_DIR_PATH)
    #eval_model(LENET_FASH_MNIST_PATH, "Lenet", IMG_FASH_MNIST_DIR_PATH)
    #eval_model(LENET_EMNIST_PATH, "Lenet", IMG_EMNIST_DIR_PATH)

    #eval_tiled_model(LENET_MNIST_PATH, "Lenet", IMG_TILED_MNIST_DIR_PATH)
    #eval_tiled_model(LENET_FASH_MNIST_PATH, "Lenet", IMG_TILED_FASH_MNIST_DIR_PATH)
    #eval_tiled_model(LENET_EMNIST_PATH, "Lenet", IMG_TILED_EMNIST_DIR_PATH)

    eval_model(LENET_MNIST_PATH, "Lenet", IMG_MNIST_GELU_DIR_PATH)


if __name__ == "__main__": 
    main()