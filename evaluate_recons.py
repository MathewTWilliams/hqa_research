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

img_root_names = [ "data_jpg",
                "data_recon_0", 
                "data_recon_1", 
                "data_recon_2", 
                "data_recon_3", 
                "data_recon_4"]

def evaluate_dataset(model, model_name, dl_test, ds_name, attack = None):
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

    conf_mat = confusion_matrix(test_labels, predictions, normalize="true")
    attack_name = attack.attack if attack != None else "None"
    add_accuracy_results(model_name, ds_name, attack_name, model.early_stopping, conf_mat.diagonal().tolist())

def eval_pickled_recons(model, model_name, column_name, pickle_path, attack = None):
    global transform

    columns = [layer_name for layer_name in LAYER_NAMES]
    columns.append("labels")
    recon_df = pd.DataFrame(data = pd.read_pickle(pickle_path), columns=columns)

    targets = recon_df["labels"].to_numpy()
    recon_df = recon_df.drop(columns="labels")

    ds_test = MyDataset(recon_df[column_name].to_numpy(), targets, transform)
    dl_test = DataLoader(ds_test, MNIST_BATCH_SIZE, shuffle=False, num_workers=4)

    evaluate_dataset(model, model_name, dl_test, column_name, attack)

    
def eval_image_recons(model, model_name, root, attack = None):
    global transform
    ds_test = ImageFolder(os.path.join(IMG_DIR_PATH, root), transform=transform)
    dl_test = DataLoader(ds_test, batch_size=MNIST_BATCH_SIZE, shuffle = False, num_workers=4)
    evaluate_dataset(model, model_name, dl_test, root, attack)
    
def main():

    global img_root_names

    # Traditional LeNet MNIST
    trad_lenet_mnist = torch.load(TRAD_LENET_MNIST_PATH)
    trad_lenet_mnist.eval()

    for root in img_root_names: 
        eval_image_recons(trad_lenet_mnist, "Trad. LeNet", root)

    fgsm_attack = torchattacks.FGSM(trad_lenet_mnist)

    for root in img_root_names:
        eval_image_recons(trad_lenet_mnist, "Trad. LeNet", root, fgsm_attack)

    del trad_lenet_mnist

    # Modern LeNet MNIST
    modern_lenet_mnist = torch.load(MOD_LENET_MNIST_PATH)
    modern_lenet_mnist.eval()

    for root in img_root_names: 
        eval_image_recons(modern_lenet_mnist, "Modern LeNet", root)

    fgsm_attack = torchattacks.FGSM(modern_lenet_mnist)

    for root in img_root_names:
        eval_image_recons(modern_lenet_mnist, "Modern LeNet", root, fgsm_attack)

    del modern_lenet_mnist

    # Traditional LeNet Fashion MNIST
    trad_lenet_fash_mnist = torch.load(TRAD_LENET_FASH_MNIST_PATH)
    trad_lenet_fash_mnist.eval()

    for root in img_root_names:
        eval_image_recons(trad_lenet_fash_mnist, "Trad. LeNet", root)

    fgsm_attack = torchattacks.FGSM(trad_lenet_fash_mnist)

    for root in img_root_names:
        eval_image_recons(trad_lenet_fash_mnist, "Trad. LeNet", root, fgsm_attack)

    del trad_lenet_fash_mnist
    
    #Modern LeNet Fashion Mnist
    mod_lenet_fash_mnist = torch.load(MOD_LENET_FASH_MNIST_PATH)
    mod_lenet_fash_mnist.eval()

    for root in img_root_names:
        eval_image_recons(mod_lenet_fash_mnist, "Modern LeNet", root)

    fgsm_attack = torchattacks.FGSM(mod_lenet_fash_mnist)

    for root in img_root_names:
        eval_image_recons(mod_lenet_fash_mnist, "Modern LeNet", root, fgsm_attack)


if __name__ == "__main__": 
    main()