# Author: Matt Williams
# Version: 11/13/2022

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from utils import ACCURACY_OUTPUT_FILE, VISUAL_DIR
import os
from utils import RECON_ROOT_NAMES

def show_and_save_graph(recon): 

    accuracy_df = pd.read_csv(ACCURACY_OUTPUT_FILE, index_col=False)

    if not os.path.exists(VISUAL_DIR): 
        os.mkdir(VISUAL_DIR)

    title = f"{recon} Lenet Accuracy Comparison"

    font_dict = {'family': 'Arial', 
                 'color' : 'darkblue', 
                 'weight' : 'normal', 
                 'size' : 16 }

    cur_df = accuracy_df[accuracy_df["Reconstruction"] == recon]

    y_positions = np.arange(len(cur_df.index))
    x_values = []
    labels = []

    for _, row_ds in cur_df.iterrows():
        cur_model = row_ds["Model"]
        cur_dataset = row_ds["Dataset"]
        cur_attack = row_ds["Attack"]
        x_values.append(row_ds["Average Accuracy"])

        cur_label = f"{cur_model} | {cur_dataset} | {cur_attack}"
        labels.append(cur_label)

    x_values = np.array(x_values)
    labels = np.array(labels)

    sorted_indexes = np.argsort(x_values)
    x_values = x_values[sorted_indexes]
    labels = labels[sorted_indexes]

    plt.figure(figsize=(22,13))
    plt.title(title, fontdict=font_dict)
    plt.xlabel("Accuracy", fontdict=font_dict)
    plt.ylabel("LeNet Configuration", fontdict = font_dict)
    plt.barh(y_positions, x_values, tick_label = labels)
    #plt.xlim([0.5, 1])
    plt.savefig(os.path.join(VISUAL_DIR, f"{recon}__accuracies.png"))
    plt.clf()

def main():
        
        for recon in RECON_ROOT_NAMES:
            show_and_save_graph(recon)

if __name__ == "__main__": 
    main()