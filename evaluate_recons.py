# Author: Matt Williams
# Version: 11/12/2022

import pandas as pd
import numpy as np
from utils import *
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
import os
import torch
from sklearn.metrics import confusion_matrix
import torchattacks

transform = transforms.Compose(
        [
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor()
        ]
    )

#ignore original cause accuracies are already calculatd for that dataset
img_root_names = [ "data_jpg",
                "data_recon_0", 
                "data_recon_1", 
                "data_recon_2", 
                "data_recon_3", 
                "data_recon_4"]

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
    add_accuracy_results(model_name, ds_name, attack_name, recon_name, avg_accuracy)


# Not currently in use
'''def eval_pickled_recons(model, model_name, column_name, pickle_path, attack = None):
    global transform

    columns = [layer_name for layer_name in LAYER_NAMES]
    columns.append("labels")
    recon_df = pd.DataFrame(data = pd.read_pickle(pickle_path), columns=columns)

    targets = recon_df["labels"].to_numpy()
    recon_df = recon_df.drop(columns="labels")

    ds_test = MyDataset(recon_df[column_name].to_numpy(), targets, transform)
    dl_test = DataLoader(ds_test, MNIST_BATCH_SIZE, shuffle=False, num_workers=4)

    evaluate_dataset(model, model_name, dl_test, column_name, attack)'''

def eval_model(model_save_path, model_name, dataset):
    global img_root_names
    global transform

    lenet_model = torch.load(model_save_path)
    lenet_model.eval()

    ds_test = ImageFolder(os.path.join(dataset, root), transform=transform)
    dl_test = DataLoader(ds_test, batch_size=MNIST_BATCH_SIZE, shuffle = False, num_workers = 4)

    for root in img_root_names:
        evaluate_dataset(lenet_model, model_name, dl_test, dataset, root)

    fgsm_attack = torchattacks.FGSM(lenet_model)

    for root in img_root_names: 
        evaluate_dataset(lenet_model, model_name, dataset, root, fgsm_attack)
    
def main():

    eval_model(TRAD_LENET_MNIST_PATH, "Trad. LeNet", IMG_MNIST_DIR_PATH)
    eval_model(TRAD_LENET_FASH_MNIST_PATH, "Trad. LeNet", IMG_FASH_MNIST_DIR_PATH)
    eval_model(TRAD_LENET_EMNIST_PATH, "Trad. LeNet", IMG_EMNIST_DIR_PATH)
    eval_model(MOD_LENET_MNIST_PATH, "Mod. LeNet", IMG_MNIST_DIR_PATH)
    eval_model(MOD_LENET_FASH_MNIST_PATH, "Mod. LeNet",  IMG_FASH_MNIST_DIR_PATH)
    eval_model(MOD_LENET_EMNIST_PATH, "Mod. LeNet", IMG_EMNIST_DIR_PATH)




if __name__ == "__main__": 
    main()