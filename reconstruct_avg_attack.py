# Author: Matt Williams
# Version: 3/20/2023

import torch
from utils import HQA_MNIST_SAVE_PATH, LENET_MNIST_PATH, device, IMG_MNIST_DIR_PATH, NUM_DATA_LOADER_WORKERS, add_accuracy_results
from torchattacks import FGSM
from torchvision import transforms
from torch.utils.data import DataLoader
import os
import numpy as np
from tqdm import tqdm
from pytorch_cnn_base import outputs_to_predictions
from numeric_image_folder import NumericImageFolder



transform = transforms.Compose(
    [
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor()
    ]
)


N_RECONSTRUCTIONS = 10

def main():

    hqa = torch.load(HQA_MNIST_SAVE_PATH)
    hqa.eval()

    lenet = torch.load(LENET_MNIST_PATH)
    lenet.eval()

    fgsm_attack = FGSM(lenet)

    root = os.path.join(IMG_MNIST_DIR_PATH, "data_original")
    ds_img_folder = NumericImageFolder(root, transform=transform)
    dl_img_folder = DataLoader(ds_img_folder, batch_size=1, shuffle = False, num_workers=NUM_DATA_LOADER_WORKERS)


    results_dict = {i: [] for i in range(10)}

    for data, label_tensor in dl_img_folder:
        cur_label = label_tensor.tolist()[0]
        cur_recons = []

        for _ in range(N_RECONSTRUCTIONS):
            data = data.to(device)
            recon = hqa.reconstruct(data).detach().cpu().squeeze()
            cur_recons.append(recon.numpy())

        cur_recons = torch.Tensor(np.array(cur_recons)).squeeze().unsqueeze(1)
        cur_labels = torch.LongTensor(label_tensor.tolist() * N_RECONSTRUCTIONS)
        cur_recons = fgsm_attack(cur_recons, cur_labels)

        cur_outputs = lenet(cur_recons.to(device)).detach().cpu()
        cur_predictions = outputs_to_predictions(cur_outputs)

        results_dict[cur_label].append(np.sum(cur_predictions == cur_label))

    avg_accuracy = 0

    for cur_label, n_correct_preds in results_dict.items(): 
            num_correct = np.sum(n_correct_preds)
            perc_correct = num_correct / (N_RECONSTRUCTIONS * len(n_correct_preds))
            weighted_avg = perc_correct * (len(num_correct) / len(ds_img_folder))

            cur_avg_accuracy += weighted_avg

    
    add_accuracy_results("Lenet(10 recons per image)", "MNIST", "data_original", fgsm_attack.attack, avg_accuracy)    

if __name__ == "__main__":
    main()