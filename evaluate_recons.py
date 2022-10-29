# Author: Matt Williams
# Version: 10/29/2022

import pandas as pd
import numpy as np
from utils import MyDataset, PICKLED_RECON_PATH, LAYER_NAMES, LENET_SAVE_PATH, device
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
import os
import torch
from sklearn.metrics import classification_report

BATCH_SIZE = 512

def get_pickled_recons():

    columns = [layer_name for layer_name in LAYER_NAMES]
    columns.append("labels")
    recon_df = pd.DataFrame(data = pd.read_pickle(PICKLED_RECON_PATH), columns=columns)

    targets = recon_df["labels"].to_numpy()
    recon_df = recon_df.drop(columns="labels")
    
    ds_dl_dict = {}

    
    #TODO make myDataset return images instead of numpy array
    for (columnName, columnData) in recon_df.items(): 
        ds_cur = MyDataset(data = columnData.to_numpy(), targets=targets)
        dl_cur = DataLoader(ds_cur)
        ds_dl_dict[columnName] = (ds_cur, dl_cur)

    

def evaluate_dataset(root):
    transform = transforms.Compose(
        [
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor()
        ]
    )

    ds_test = ImageFolder(os.path.join(os.getcwd(), root), transform=transform)
    dl_test = DataLoader(ds_test, batch_size=BATCH_SIZE, shuffle = False, num_workers=4)

    lenet = torch.load(LENET_SAVE_PATH)
    lenet.eval()
    all_outputs = torch.Tensor().to(device)
    test_labels = []

    with torch.no_grad():
        for data, labels in dl_test:
            cur_output = lenet(data.to(device))
            all_outputs = torch.cat((all_outputs, cur_output), 0)
            test_labels.extend(labels.tolist())
    
    softmax_probs = torch.exp(all_outputs).cpu()
    predictions = np.argmax(softmax_probs, axis = -1)

    class_report = classification_report(test_labels, predictions, output_dict=True)

    print("Classification results for:", root)
    for key in class_report:
        print(key,":", class_report[key])
    print("==========================================================")
        
def main():

    root_names = [ "data_original",
                    "data_jpg",
                    "data_recon_0", 
                    "data_recon_1", 
                    "data_recon_2", 
                    "data_recon_3", 
                    "data_recon_4"]

    for root in root_names: 
        evaluate_dataset(root)


if __name__ == "__main__": 
    main()