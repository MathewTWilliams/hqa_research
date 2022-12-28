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

img_folder_transform = transforms.Compose(
        [
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor()
        ]
    )

def evaluate_dataset(model, model_name, dl_test, ds_name, recon_name, attack = None):
    all_outputs = torch.Tensor().to(device)
    test_labels = []

    for data, labels in dl_test:
        if attack != None: 
            data = attack(data, labels)
        cur_output = model(data.to(device))
        all_outputs = torch.cat((all_outputs, cur_output), 0)
        test_labels.extend(labels.tolist())
    
    softmax_probs = torch.exp(all_outputs).detach().cpu().numpy()
    predictions = np.argmax(softmax_probs, axis = -1)

    diag_accuracies = confusion_matrix(test_labels, predictions, normalize="true").diagonal().tolist()
    avg_accuracy = np.sum(diag_accuracies) / len(diag_accuracies)
    attack_name = attack.attack if attack != None else "None"
    add_accuracy_results(model_name, ds_name, recon_name, attack_name, avg_accuracy)

def eval_model(model_save_path, model_name, dataset):
    global img_folder_transform

    lenet_model = torch.load(model_save_path)
    lenet_model.eval()

    ds_name = dataset.split("\\")[-1]

    for root in tqdm(RECON_ROOT_NAMES):
        ds_test = NumericImageFolder(os.path.join(dataset, root), transform=img_folder_transform)
        dl_test = DataLoader(ds_test, batch_size=MNIST_BATCH_SIZE, shuffle = False, num_workers = 4)
        evaluate_dataset(lenet_model, model_name, dl_test, ds_name, root)

    fgsm_attack = torchattacks.FGSM(lenet_model)

    for root in tqdm(RECON_ROOT_NAMES):
        ds_test = NumericImageFolder(os.path.join(dataset, root), transform=img_folder_transform)
        dl_test = DataLoader(ds_test, batch_size=MNIST_BATCH_SIZE, shuffle = False, num_workers = 4)
        evaluate_dataset(lenet_model, model_name, dl_test, ds_name, root, fgsm_attack)
    
def main():

    eval_model(LENET_MNIST_PATH, "Lenet", IMG_MNIST_DIR_PATH)
    eval_model(LENET_FASH_MNIST_PATH, "Lenet", IMG_FASH_MNIST_DIR_PATH)
    eval_model(LENET_EMNIST_PATH, "Lenet", IMG_EMNIST_DIR_PATH)



if __name__ == "__main__": 
    main()