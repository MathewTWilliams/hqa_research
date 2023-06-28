from save_load_json import load_from_json_file
import numpy as np
import matplotlib.pyplot as plt
from utils import JSON_DIR_PATH, RECON_REPEAT_VIS_DIR
import os
import matplotlib.colors as mcolors

def make_recons_visual(filename):
    file_path = os.path.join(JSON_DIR_PATH, filename)
    json_dict = load_from_json_file(file_path)
    
    x_ticks = list(range(11))
    legend_labels = json_dict.keys()
    font_dict = {'family': 'Arial', 
                 'color' : 'darkblue', 
                 'weight' : 'normal', 
                 'size' : 32}


    label_freq_count = {}

    for label, freq_dict in json_dict.items():
        freq_values = []
        for _, freq in freq_dict.items():
            freq_values.append(freq)

        label_freq_count[label] = freq_values

    fig, ax1 = plt.subplots()
    fig.set_figwidth(18)
    fig.set_figheight(10)
    ax1.set_title("Correct Classification", **font_dict)
    ax1.set_xlabel("n out of 10", **font_dict, labelpad = 10)
    ax1.set_ylabel("frequency", **font_dict, labelpad=30)
    ax1.tick_params(axis = "both", which = "major", labelsize = 30)


    colors = list(mcolors.TABLEAU_COLORS.keys())


    for label, freq_values in label_freq_count.items():
        ax1.plot(x_ticks, freq_values, color = colors.pop(0), linewidth = 7)


    ax1.legend(labels = legend_labels, loc = 0, fontsize = 24, borderpad = 0.2, labelspacing = 0.1, borderaxespad = 0.1, frameon = False)


    file_name = f"{filename.split('.')[0]}.png"

    plt.savefig(os.path.join(RECON_REPEAT_VIS_DIR, file_name))
    plt.close("all")

if __name__ == "__main__":
     
    if not os.path.isdir(RECON_REPEAT_VIS_DIR):
        os.mkdir(RECON_REPEAT_VIS_DIR)



    for file in os.listdir(JSON_DIR_PATH):
        make_recons_visual(file)
        