# Author: Matt Williams
# Version: 3/20/2023

import torch
from utils import HQA_MNIST_SAVE_PATH, LENET_MNIST_PATH, device, IMG_MNIST_DIR_PATH, NUM_DATA_LOADER_WORKERS, IMG_FOLDER_TRANFORM
from torchattacks import FGSM
import os
import numpy as np
from tqdm import tqdm
from pytorch_cnn_base import outputs_to_predictions
from numeric_image_folder import NumericImageFolder
from torchattacks import FGSM
from save_load_json import save_to_json_file

N_RECONSTRUCTIONS = 20

def run_reconstruct_avg(model, save_file_name, ds_img_folder = None, ds_idxs = None, layer_num = 0, attack_img = False, attack_recons = False):
    
    hqa = torch.load(HQA_MNIST_SAVE_PATH)
    hqa.eval()
    hqa_layer_n = hqa[layer_num]

    if ds_img_folder is None: 
        root = os.path.join(IMG_MNIST_DIR_PATH, "data_original")
        ds_img_folder = NumericImageFolder(root, transform=IMG_FOLDER_TRANFORM)

    count_map = {num: {count: 0 for count in range(N_RECONSTRUCTIONS+1)} for num in range(11)}

    if ds_idxs is None: 
        ds_idxs = range(len(ds_img_folder))

    fgsm_attack = None
    if attack_img or attack_recons: 
        fgsm_attack = FGSM(model)

    for idx in ds_idxs: 
        img, label = ds_img_folder[idx]
        if attack_img: 
            img = fgsm_attack(img.unsqueeze(0), torch.LongTensor([label])).squeeze(0)#.detach().cpu()
        cur_recons = []

        for _ in range(N_RECONSTRUCTIONS):
            recon = hqa_layer_n.reconstruct(img).squeeze(0)#.detach().cpu()
            cur_recons.append(recon.numpy())

        cur_recons = torch.Tensor(np.array(cur_recons)).squeeze().unsqueeze(1)
        cur_labels = torch.LongTensor([label] * N_RECONSTRUCTIONS)

        if attack_recons: 
            cur_recons = fgsm_attack(cur_recons, cur_labels)

        cur_outputs = model(cur_recons).detach().cpu() #cur_recons.to(device)
        cur_predictions = outputs_to_predictions(cur_outputs)
    
        n_correct = np.sum(cur_predictions == cur_labels)
        count_map[label][n_correct] += 1

    save_to_json_file(count_map, save_file_name)


def run_dict_reconstruct_avg(model, save_file_name, img_map, layer_num = 0, attack_img = False, attack_recons = False): 

    hqa = torch.load(HQA_MNIST_SAVE_PATH)
    hqa.eval()
    hqa_layer_n = hqa[layer_num]

    count_map = {num: {count: 0 for count in range(N_RECONSTRUCTIONS+1)} for num in range(11)}

    fgsm_attack = None
    if attack_img or attack_recons: 
        fgsm_attack = FGSM(model)

    for label, imgs in img_map:
        for img in imgs: 

            if attack_img: 
                img = fgsm_attack(img.unsqueeze(0), torch.LongTensor([label])).squeeze(0)#.detach().cpu()

            for _ in range(N_RECONSTRUCTIONS): 
                recon = hqa_layer_n.reconstruct(img).squeeze(0)
                cur_recons.append(recon.numpy())

        cur_recons = torch.Tensor(np.array(cur_recons)).squeeze().unsqueeze(1)
        cur_labels = torch.LongTensor([label] * N_RECONSTRUCTIONS)

        if attack_recons: 
            cur_recons = fgsm_attack(cur_recons, cur_labels)

        cur_outputs = model(cur_recons).detach().cpu() #cur_recons.to(device)
        cur_predictions = outputs_to_predictions(cur_outputs)
    
        n_correct = np.sum(cur_predictions == cur_labels)
        count_map[label][n_correct] += 1

    save_to_json_file(count_map, save_file_name)

if __name__ == "__main__":
    run_reconstruct_avg("hqa_0_20_recons_full_dataset_normal.json")
    run_reconstruct_avg("hqa_0_20_recons_full_dataset_atk_img.json", attack_img=True)
    run_reconstruct_avg("hqa_0_20_recons_full_dataset_atk_recons.json", attack_recons=True)
    run_reconstruct_avg("hqa_0_20_recons_full_dataset_atk_both.json", attack_img=True, attack_recons=True)

    run_reconstruct_avg("hqa_4_20_recons_full_dataset_normal.json", hqa_layer=4)
    run_reconstruct_avg("hqa_4_20_recons_full_dataset_atk_img.json", hqa_layer = 4, attack_img=True)
    run_reconstruct_avg("hqa_4_20_recons_full_dataset_atk_recons.json", hqa_layer = 4, attack_recons=True)
    run_reconstruct_avg("hqa_4_20_recons_full_dataset_atk_both.json", hqa_layer = 4, attack_img=True, attack_recons=True)