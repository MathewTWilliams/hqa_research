# Author: Matt Williams
# Version: 11/18/2022


import torch
from utils import HQA_SAVE_PATH, IMG_DIR_PATH, LENET_SAVE_PATH, device, EARLY_LENET_SAVE_PATH
from utils import add_accuracy_results
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
import os
import torchattacks 
from hqa import *
import numpy as np
from LeNet import LeNet_5
from sklearn.metrics import confusion_matrix
from tqdm import tqdm

transform = transforms.Compose(
    [
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor()
    ]
)

N_RECONSTRUCTIONS = 14

def main(model):
    global transform

    hqa = torch.load(HQA_SAVE_PATH)
    hqa.eval()


    root = os.path.join(IMG_DIR_PATH, "data_original")
    ds_img_folder_test = ImageFolder(root, transform=transform)
    dl_img_folder_test = DataLoader(ds_img_folder_test, shuffle=False, num_workers=4)

    model.eval()
    fgsm_attack = torchattacks.FGSM(model)

    results_dict = {i : [] for i in range(10)}

    for data, label_tensor in tqdm(dl_img_folder_test):
        cur_label = label_tensor.tolist()[0]
        cur_recons = []
        for _ in range(N_RECONSTRUCTIONS):
            data = data.to(device)
            recon = hqa.reconstruct(data).detach().cpu().squeeze()
            cur_recons.append(recon.numpy())

        cur_recons = torch.Tensor(np.array(cur_recons)).squeeze().unsqueeze(1)
        cur_labels = torch.LongTensor(label_tensor.tolist() * N_RECONSTRUCTIONS)
        cur_recons = fgsm_attack(cur_recons, cur_labels)

        cur_outputs = model(cur_recons.to(device))
        softmax_probs = torch.exp(cur_outputs).detach().cpu().numpy()
        predictions = np.argmax(softmax_probs, axis = -1)
        conf_mat = confusion_matrix(cur_labels.tolist(), predictions, normalize = "true", labels = [i for i in range(10)])
        results_dict[cur_label] = conf_mat[cur_label][cur_label]

    '''mean_accuracies = []
    for key in results_dict:
        mean_accuracies.append(np.mean(results_dict[key]))

    add_accuracy_results("LeNet(14 recons/datapoint)", "data_original", fgsm_attack.attack, model.early_stopping, mean_accuracies)'''

if __name__ == "__main__": 
    main(torch.load(EARLY_LENET_SAVE_PATH))
    #main(torch.load(LENET_SAVE_PATH))