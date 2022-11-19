# Author: Matt Williams
# Version: 11/13/2022

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from utils import ACCURACY_OUTPUT_FILE, VISUAL_DIR
import os


def show_and_save_graph(acc_df, col): 
    if not os.path.exists(VISUAL_DIR): 
        os.mkdir(VISUAL_DIR)

    insert = col if col == "Average" else f"Number {col}"
    title = f"LeNet {insert} Accuracy Comparison"

    font_dict = {'family': 'Arial', 
                 'color' : 'darkblue', 
                 'weight' : 'normal', 
                 'size' : 16 }

    y_positions = np.arange(len(acc_df.index))
    x_values = []
    labels = []

    for _, row_ds in acc_df.iterrows():
        x_values.append(row_ds[col])
        cur_dataset = row_ds["Dataset"].replace("data_","")
        cur_attack = row_ds["Attack"]
        cur_es_val = row_ds["Early Stopping"]
        
        cur_es_val = "E.S." if cur_es_val else "No E.S."

        cur_label = f"{cur_dataset} | {cur_attack} | {cur_es_val}"
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
    plt.xlim([0.5, 1])
    plt.savefig(os.path.join(VISUAL_DIR, f"{col}_accuracies.png"))
    plt.clf()

def main():

    accuracy_df = pd.read_csv(ACCURACY_OUTPUT_FILE, index_col=False)
    scores_starting_index = 4

    for col in accuracy_df.columns[scores_starting_index:]:
        show_and_save_graph(accuracy_df, col)


if __name__ == "__main__": 
    main()