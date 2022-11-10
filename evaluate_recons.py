# Author: Matt Williams
# Version: 10/29/2022

import pandas as pd
import numpy as np
from utils import MyDataset, PICKLED_RECON_PATH, LAYER_NAMES, LENET_SAVE_PATH, device, IMG_DIR_PATH
from utils import add_mnist_accuracies
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
import os
import torch
from sklearn.metrics import confusion_matrix
import torchattacks


BATCH_SIZE = 512

transform = transforms.Compose(
        [
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor()
        ]
    )

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
    add_mnist_accuracies(model_name, ds_name, attack_name, conf_mat.diagonal().tolist())

def eval_pickled_recons(model, model_name, column_name, attack = None):
    global transform

    columns = [layer_name for layer_name in LAYER_NAMES]
    columns.append("labels")
    recon_df = pd.DataFrame(data = pd.read_pickle(PICKLED_RECON_PATH), columns=columns)

    targets = recon_df["labels"].to_numpy()
    recon_df = recon_df.drop(columns="labels")

    ds_test = MyDataset(recon_df[column_name].to_numpy(), targets, transform)
    dl_test = DataLoader(ds_test, BATCH_SIZE, shuffle=False, num_workers=4)

    evaluate_dataset(model, model_name, dl_test, column_name, attack)

    
def eval_image_recons(model, model_name, root, attack = None):
    global transform
    ds_test = ImageFolder(os.path.join(IMG_DIR_PATH, root), transform=transform)
    dl_test = DataLoader(ds_test, batch_size=BATCH_SIZE, shuffle = False, num_workers=4)
    evaluate_dataset(model, model_name, dl_test, root, attack)
    

        
def main():
    lenet = torch.load(LENET_SAVE_PATH)
    lenet.eval()

    img_root_names = ["data_original",
                    "data_jpg",
                    "data_recon_0", 
                    "data_recon_1", 
                    "data_recon_2", 
                    "data_recon_3", 
                    "data_recon_4"]

    #for layer in LAYER_NAMES:
    #    eval_pickled_recons(lenet, "LeNet-5", layer)

    for root in img_root_names: 
        eval_image_recons(lenet, "LeNet-5", root)
        

    fgsm_attack = torchattacks.FGSM(lenet)
    
    for root in img_root_names: 
        eval_image_recons(lenet, "LeNet-5", root, fgsm_attack)

if __name__ == "__main__": 
    main()