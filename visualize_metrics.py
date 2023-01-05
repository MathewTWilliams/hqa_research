# Author: Matt Williams
# Version: 11/13/2022

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from utils import ACCURACY_OUTPUT_FILE, VISUAL_DIR
import os
from utils import RECON_ROOT_NAMES

def make_and_save_line_graph(dataset, model_name): 

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



    reg_y_values = []
    atk_y_values = []
    x_ticks = []

    for _, row_ds in cur_df.iterrows():
        cur_recon = row_ds["Reconstruction"]
        cur_attack = row_ds["Attack"]
        if cur_attack == "None":
            reg_y_values.append(row_ds["Average Accuracy"])
            x_ticks.append(cur_recon)
        else:
            atk_y_values.append(row_ds["Average Accuracy"])

    plt.figure(figsize=(20,10))
    plt.title(title, fontdict=font_dict)
    plt.xlabel("Reconstruction Layer", fontdict=font_dict)
    plt.ylabel("Accuracy", fontdict=font_dict)
    plt.plot(x_ticks, reg_y_values, label = "Regular")
    plt.plot(x_ticks, atk_y_values, label = "FGSM Attack")
    plt.legend()
    plt.savefig(os.path.join(VISUAL_DIR, f"{model_name}_{dataset}_accuracies.png"))
    plt.clf()

def main():
        
    datasets = ["MNIST", "Fashion_MNIST", "EMNIST", "MNIST_GELU",
                "Tiled_MNIST", "Tiled_Fashion_MNIST", "Tiled_EMNIST"]
    for dataset in datasets:
        make_and_save_line_graph(dataset, "Lenet")

if __name__ == "__main__": 
    main()