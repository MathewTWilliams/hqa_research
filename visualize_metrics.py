# Author: Matt Williams
# Version: 11/13/2022

import matplotlib.pyplot as plt
import pandas as pd
from utils import *
import os
import re
import numpy as np


def _read_log_files(hqa_model_name, num_layer, read_value):
    file_name = f"{hqa_model_name}_l{num_layer}.log"
    file_path = os.path.join(LOG_DIR, file_name)
    output = []
    with open(file_path, mode="r", encoding="utf-8") as file:
        for line in file.readlines():
            regex = f"\s{read_value}.*?,\s"
            value = re.search(regex, line).group().strip()[:-1].split("=")[-1]
            output.append(float(value))
    
    return output


def _get_jpg_compression(dataset):
    jpg_sizes = []
    reg_sizes = []

    root_path = os.path.join(IMG_DIR_PATH, dataset, "data_jpg")
    for sub_dir in os.listdir(root_path):
        sub_dir_path = os.path.join(root_path, sub_dir)
        for img_file in os.listdir(sub_dir_path):
            img_path = os.path.join(sub_dir_path, img_file)
            jpg_sizes.append(os.path.getsize(img_path))
    
    orig_root_path = os.path.join(IMG_DIR_PATH, dataset, "data_original")
    for sub_dir in os.listdir(root_path):
        sub_dir_path = os.path.join(orig_root_path, sub_dir)
        for img_file in os.listdir(sub_dir_path):
            img_path = os.path.join(sub_dir_path, img_file)
            reg_sizes.append(os.path.getsize(img_path))
    

    return np.sum(np.divide(jpg_sizes, reg_sizes)) / len(reg_sizes)

   


def make_and_save_line_graph(dataset, model_name, second_y_ax = None, hqa_model_name = None): 

    accuracy_df = pd.read_csv(ACCURACY_OUTPUT_FILE, index_col=False)

    if not os.path.exists(VISUAL_DIR): 
        os.mkdir(VISUAL_DIR)

    title = f"{model_name} Accuracy Comparison on {dataset}"

    font_dict = {'family': 'Arial', 
                 'color' : 'darkblue', 
                 'weight' : 'normal', 
                 'size' : 16 }

    cur_df = accuracy_df[(accuracy_df["Dataset"] == dataset) & \
                        (accuracy_df["Model"] == model_name)]


    reg_acc_values = []
    atk_acc_values = []
    second_y_values = []
    x_ticks = []

    for _, row_ds in cur_df.iterrows():
        cur_recon = row_ds["Reconstruction"]
        cur_attack = row_ds["Attack"]

        if cur_attack == "None":
            reg_acc_values.append(row_ds["Average Accuracy"])
            x_ticks.append(cur_recon)

            if second_y_ax is not None:
                if cur_recon == "data_original":
                    second_y_values.append(1)
                elif cur_recon == "data_jpg":
                    second_y_values.append(_get_jpg_compression(dataset))
                else: 
                    num_layer = cur_recon[-1]
                    log_values = _read_log_files(hqa_model_name, num_layer, second_y_ax)
                    second_y_values.append(log_values[-1])
        else:
            atk_acc_values.append(row_ds["Average Accuracy"])

    print(second_y_values)
    #TODO fix second y axis order and legend
    fig, ax1 = plt.subplots()
    plt.figure(figsize=(20,10))
    plt.title(title, fontdict=font_dict)
    ax1 = plt.gca()
    ax1.set_xlabel("Reconstruction Layer", **font_dict)
    ax1.set_ylabel("Accuracy", **font_dict)
    ax1.plot(x_ticks, reg_acc_values, label = "Regular", color = "blue")
    ax1.plot(x_ticks, atk_acc_values, label = "FGSM Attack", color = "orange")
    ax1.legend(loc=3)

    if second_y_ax is not None:
        ax2 = ax1.twinx()
        ax2.set_ylabel(second_y_ax, **font_dict, rotation = 270)
        ax2.plot(x_ticks, second_y_values, label = "2nd Regular", color = "purple")
        ax2.legend(loc=0)
    
    plt.savefig(os.path.join(VISUAL_DIR, f"{model_name}_{dataset}_accuracies.png"))
    plt.clf()

def main():

    ds_hqa_map = {"EMNIST" : HQA_EMNIST_MODEL_NAME,
                "Fashion_MNIST" : HQA_FASH_MNIST_MODEL_NAME,
                "MNIST" : HQA_MNIST_MODEL_NAME,
                "MNIST_GELU" : HQA_MNIST_GELU_MODEL_NAME,
                "Tiled_EMNIST" : HQA_TILED_EMNIST_MODEL_NAME,
                "Tiled_Fashion_MNIST" : HQA_TILED_FASH_MNIST_MODEL_NAME,
                "Tiled_MNIST" : HQA_TILED_MNIST_MODEL_NAME}

    datasets = os.listdir(IMG_DIR_PATH)
    for dataset in datasets:
        if dataset == "MNIST_FFT":
            continue
        make_and_save_line_graph(dataset, "Lenet", "rate", ds_hqa_map[dataset])
        

if __name__ == "__main__": 
    main()