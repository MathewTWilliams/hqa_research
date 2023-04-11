# Author: Matt Williams
# Version: 3/20/2023

import torch
from utils import HQA_MNIST_SAVE_PATH, LENET_MNIST_PATH, device, IMG_MNIST_DIR_PATH, NUM_DATA_LOADER_WORKERS, \
    IMG_FOLDER_TRANFORM, RECONS_EXPERIMENT_DIR, save_img
from torchattacks import FGSM
import os
import numpy as np
from tqdm import tqdm
from pytorch_cnn_base import outputs_to_predictions
from numeric_image_folder import NumericImageFolder
from torchattacks import FGSM
from save_load_json import save_to_json_file

N_RECONSTRUCTIONS = 10

def run_reconstruct_avg(model, save_file_name, ds_img_folder = None, ds_idxs = None, layer_num = 0, attack_img = False, attack_recons = False):
    
    if not os.path.exists(RECONS_EXPERIMENT_DIR): 
        os.mkdir(RECONS_EXPERIMENT_DIR)

    sub_dir_path = os.path.join(RECONS_EXPERIMENT_DIR, save_file_name.split(".")[0])
    if not os.path.exists(sub_dir_path): 
        os.mkdir(sub_dir_path)

    hqa = torch.load(HQA_MNIST_SAVE_PATH)
    hqa.eval()
    hqa_layer_n = hqa[layer_num]

    if ds_img_folder is None: 
        root = os.path.join(IMG_MNIST_DIR_PATH, "data_original")
        ds_img_folder = NumericImageFolder(root, transform=IMG_FOLDER_TRANFORM)

    count_map = {num: {count: 0 for count in range(N_RECONSTRUCTIONS+1)} for num in range(10)}

    if ds_idxs is None: 
        ds_idxs = range(len(ds_img_folder))

    fgsm_attack = None
    if attack_img or attack_recons: 
        fgsm_attack = FGSM(model)

    for i, idx in enumerate(ds_idxs):
        img, label = ds_img_folder[idx]
        save_img(img.squeeze(0), label, sub_dir_path, i, False, 0, "original")

        if attack_img: 
            img = fgsm_attack(img.unsqueeze(0), torch.LongTensor([label])).squeeze(0)#.detach().cpu()
            save_img(img.squeeze(0), label, sub_dir_path, i, False, 0, "atk_original")

        cur_recons = []

        for j in range(N_RECONSTRUCTIONS):
            recon = hqa_layer_n.reconstruct(img.to(device).unsqueeze(0)).squeeze(0).detach().cpu()
            save_img(recon.squeeze(0), label, sub_dir_path, i, False, 0, f"recon_{layer_num}_{j}")
            if attack_recons: 
                recon = fgsm_attack(recon.unsqueeze(0), torch.LongTensor([label])).squeeze(0)#.detach().cpu()
                save_img(recon.squeeze(0), label, sub_dir_path, i, False, 0, f"atk_recon_{layer_num}_{j}")
            cur_recons.append(recon.numpy())

        cur_recons = torch.Tensor(np.array(cur_recons)).squeeze().unsqueeze(1)
        cur_labels = torch.LongTensor([label] * N_RECONSTRUCTIONS)

        cur_outputs = model(cur_recons.to(device)).detach().cpu()
        cur_predictions = outputs_to_predictions(cur_outputs)
    
        n_correct = np.sum(cur_predictions == cur_labels.numpy())
        count_map[label][n_correct] += 1

    save_to_json_file(count_map, save_file_name)


def run_dict_reconstruct_avg(model, save_file_name, img_map, layer_num = 0, attack_img = False, attack_recons = False): 

    if not os.path.exists(RECONS_EXPERIMENT_DIR): 
        os.mkdir(RECONS_EXPERIMENT_DIR)

    sub_dir_path = os.path.join(RECONS_EXPERIMENT_DIR, save_file_name.split(".")[0])
    if not os.path.exists(sub_dir_path): 
        os.mkdir(sub_dir_path)

    hqa = torch.load(HQA_MNIST_SAVE_PATH)
    hqa.eval()
    hqa_layer_n = hqa[layer_num]

    count_map = {num: {count: 0 for count in range(N_RECONSTRUCTIONS+1)} for num in range(10)}

    fgsm_attack = None
    if attack_img or attack_recons: 
        fgsm_attack = FGSM(model)

    for label, imgs in img_map.items():
        for i, img in enumerate(imgs): 
            save_img(img.squeeze(0), label, sub_dir_path, i, False, 0, "original")
            if attack_img: 
                img = fgsm_attack(img.unsqueeze(0), torch.LongTensor([label])).squeeze(0)#.detach().cpu()
                save_img(img.squeeze(0), label, sub_dir_path, i, False, 0, "atk_original")

            cur_recons = []
            
            for j in range(N_RECONSTRUCTIONS): 
                recon = hqa_layer_n.reconstruct(img.to(device).unsqueeze(0)).squeeze(0).detach().cpu()
                save_img(recon.squeeze(0), label, sub_dir_path, i, False, 0, f"recon_{layer_num}_{j}")
                if attack_recons: 
                    recon = fgsm_attack(recon.unsqueeze(0), torch.LongTensor([label])).squeeze(0)#.detach().cpu()
                    save_img(img.squeeze(0), label, sub_dir_path, i, False, 0, f"atk_recon_{layer_num}_{j}")
                cur_recons.append(recon.numpy())

            cur_recons = torch.Tensor(np.array(cur_recons)).squeeze().unsqueeze(1)
            cur_labels = torch.LongTensor([label] * N_RECONSTRUCTIONS)

            cur_outputs = model(cur_recons.to(device)).detach().cpu()
            cur_predictions = outputs_to_predictions(cur_outputs)
        
            n_correct = np.sum(cur_predictions == cur_labels.numpy())
            count_map[label][n_correct] += 1

    save_to_json_file(count_map, save_file_name)
