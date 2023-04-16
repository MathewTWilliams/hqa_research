# Author: Matt Williams
# Version: 04/15/2022

import matplotlib.pyplot as plt
import pandas as pd
from utils import *
import os
import numpy as np
import matplotlib.colors as mcolors

RATE_DICT = {
    "data_original" : 1.0,
    "data_recon_0" : 0.25,
    "data_recon_1" : 0.0625,
    "data_recon_2" : 1.5625e-2,
    "data_recon_3" : 3.90625e-3,
    "data_recon_4" : 9.765625e-4
}

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

   
def make_and_save_accuracy_plot(model_dataset_tuples, file_name, second_y_ax = None, dataset_second_y_calc = None): 

    accuracy_df = pd.read_csv(ACCURACY_OUTPUT_FILE, index_col=False)

    if not os.path.isdir(ACCURACY_VIS_DIR): 
        os.makedirs(ACCURACY_VIS_DIR)

    font_dict = {'family': 'Arial', 
                 'color' : 'darkblue', 
                 'weight' : 'normal', 
                 'size' : 32}
    
    has_tiled = False
    for _, dataset in model_dataset_tuples:
        if dataset.find("Tiled") != -1: 
            has_tiled = True
            break


    x_ticks = [recon.split("_")[-1] for recon in RECON_ROOT_NAMES]
    if has_tiled: 
        x_ticks = x_ticks[:-1]

    reg_acc_values = []
    atk_acc_values = []

    for model_name, dataset in model_dataset_tuples:  

        cur_df = accuracy_df[(accuracy_df["Dataset"] == dataset) & \
                            (accuracy_df["Model"] == model_name)]
        
        cur_reg_acc_values = []
        cur_atk_acc_values = []

        #grab accuracies
        for _, row_ds in cur_df.iterrows():
            cur_recon = row_ds["Reconstruction"]
            cur_attack = row_ds["Attack"]

            # ignore tiled data sets that combine different hqa layers
            if cur_recon.find("&") != -1:
                continue

            if cur_attack == "None":
                cur_reg_acc_values.append(row_ds["Average Accuracy"])
            else:
                cur_atk_acc_values.append(row_ds["Average Accuracy"])

        reg_acc_values.append(cur_reg_acc_values)
        atk_acc_values.append(cur_atk_acc_values)

    second_y_values = []
    make_second_y = (second_y_ax is not None and dataset_second_y_calc is not None)

    if make_second_y:
        recon_names = RECON_ROOT_NAMES if not has_tiled else RECON_ROOT_NAMES[:-1]
        for recon in recon_names:

            if recon == "data_jpg":
                if dataset_second_y_calc == "MNIST Recons": 
                    second_y_values.append(_get_jpg_compression("MNIST"))
                else: 
                    second_y_values.append(_get_jpg_compression(dataset))

            else: 
                second_y_values.append(RATE_DICT[recon])


    _, ax1 = plt.subplots()
    plt.figure(figsize=(18,10))
    ax1 = plt.gca()
    ax1.set_xlabel("Reconstruction Layer", **font_dict, labelpad = 10)
    ax1.set_ylabel("Accuracy", **font_dict, labelpad = 30)
    ax1.set_yticks(ticks = [0.75, 0.8, 0.85, 0.9, 0.95, 1.0])
    ax1.tick_params(axis = "both", which = "major", labelsize=30)
    all_plots = None
    labels = []
    colors = list(mcolors.TABLEAU_COLORS.keys())

    for i, (cur_reg_acc_values, cur_atk_acc_values) in enumerate(zip(reg_acc_values, atk_acc_values)):
        model, dataset = model_dataset_tuples[i] 
        plot_1 = ax1.plot(x_ticks, cur_reg_acc_values, color = colors.pop(0), linewidth = 7, marker = "s", markersize = 11)
        plot_2 = ax1.plot(x_ticks, cur_atk_acc_values, color = colors.pop(0), linewidth = 7, marker = "s", markersize = 11)
        if all_plots is None: 
            all_plots = plot_1 + plot_2
        else: 
            all_plots += plot_1 + plot_2
        

        labels.append(f"{model}: Org. {dataset}")
        labels.append(f"{model}: Atk. {dataset}")

    if make_second_y:
        ax2 = ax1.twinx()
        ax2.tick_params(axis="y", which = "major", labelsize = 30)
        ax2.set_ylabel(second_y_ax, **font_dict, rotation = 270, labelpad = 35)
        ax2.set_yticks(ticks = [0.0, 0.25, 0.5, 0.75, 1.0])
        plot_3 = ax2.plot(x_ticks, second_y_values, color = colors.pop(0), linewidth = 7, marker = "s", markersize = 11)
        all_plots += plot_3
        labels.append("Compression Rate")

    ax1.legend(all_plots, labels, loc = 0, fontsize = 16)
    plt.savefig(os.path.join(ACCURACY_VIS_DIR, file_name))
    plt.clf()

def main():

    '''valid_hqa_datasets = ["EMNIST",
                        "Fashion_MNIST", 
                        "MNIST", 
                        "Tiled_EMNIST", 
                        "Tiled_Fashion_MNIST",
                        "Tiled_MNIST",
                        "MNIST Recons"]

    for dataset in valid_hqa_datasets:
        if dataset == "MNIST Recons": 
            make_and_save_accuracy_plot([("Lenet (Recons)", dataset)], f"Lenet (Recons)_{dataset}_accuracies.png" ,"rate", dataset)
        else: 
            make_and_save_accuracy_plot([("Lenet", dataset)], f"Lenet_{dataset}_accuracies.png" ,"rate", dataset)

        if dataset == "MNIST":
            make_and_save_accuracy_plot([("Lenet (Adversarial)", dataset)], f"Lenet (Adversarial)_{dataset}_accuracies.png" ,"rate", dataset)'''

    model_dataset_tuples = [
        ("Lenet", "MNIST"), 
        ("Lenet (Recons)", "MNIST Recons"), 
        ("Lenet (Adversarial)", "MNIST")
    ]

    make_and_save_accuracy_plot(model_dataset_tuples, "combined_accuracies.png", "rate", "MNIST")

if __name__ == "__main__": 
    main()