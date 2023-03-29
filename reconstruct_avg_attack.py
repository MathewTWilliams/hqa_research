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
from torchattacks import FGSM

transform = transforms.Compose(
    [
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor()
    ]
)

N_RECONSTRUCTIONS = 10

def main(run_attack = False):

    hqa = torch.load(HQA_MNIST_SAVE_PATH)
    hqa.eval()

    lenet = torch.load(LENET_MNIST_PATH)
    lenet.eval()

    fgsm_attack = None
    if run_attack: 
        fgsm_attack = FGSM(lenet)

    root = os.path.join(IMG_MNIST_DIR_PATH, "data_original")
    ds_img_folder = NumericImageFolder(root, transform=transform)
    dl_img_folder = DataLoader(ds_img_folder, batch_size=1, shuffle = True, num_workers=NUM_DATA_LOADER_WORKERS)


    num_correct = 0
    count_map = {num: 0 for num in range(11)}

    for i, (data, label_tensor) in enumerate(dl_img_folder):
        cur_label = label_tensor.tolist()[0]
        cur_recons = []

        for _ in range(N_RECONSTRUCTIONS):
            data = data.to(device)
            recon = hqa.reconstruct(data).detach().cpu().squeeze()
            cur_recons.append(recon.numpy())

        cur_recons = torch.Tensor(np.array(cur_recons)).squeeze().unsqueeze(1)
        cur_labels = torch.LongTensor(label_tensor.tolist() * N_RECONSTRUCTIONS)
        if run_attack:
            cur_recons = fgsm_attack(cur_recons, cur_labels)

        cur_outputs = lenet(cur_recons.to(device)).detach().cpu()
        cur_predictions = outputs_to_predictions(cur_outputs)

        num_correct += np.sum(cur_predictions == cur_label)
        count_map[np.sum(cur_predictions == cur_label)] += 1
        attacked_img = "Attacked" if run_attack else "Normal"
        print(f"{attacked_img} current image: {cur_label}, Number of reconstructions correct: {np.sum(cur_predictions == cur_label)}")
        if i == 100:
            for key,item in count_map.items(): 
                print(f"{key} : {item}")

            break


    #accuracy = num_correct / (N_RECONSTRUCTIONS * len(ds_img_folder))            
    #attack_name = fgsm_attack.attack if run_attack else "None"

    #add_accuracy_results("Lenet(10 recons per image)", "MNIST", "data_original", attack_name, accuracy)    

if __name__ == "__main__":
    main(run_attack=False)
    main(run_attack=True)