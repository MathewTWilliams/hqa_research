# Author: Matt Williams
# Version: 1/31/2023

import pandas as pd
import matplotlib.pyplot as plt
from utils import *
import os
import numpy as np

"""
Made entropies for the following: 
- regular correct predictions (image and CNN output)
- attacked incorrect predictions (image and CNN output)

- 2 plots per visual
-- One for regular correct predictions and attacked incorrect predictions
- 2 of these 2 plot visuals are made
-- one for images and one for CNN output

- Need to make these 4 plots (2 visualizations) for each reconstruction type
"""
COLOR_DICT = {
    0 : "red", 
    1 : "yellow",
    2 : "green",
    3 : "purple",
    4 : "pink", 
    5 : "magenta",
    6 : "blue",
    7 : "black",
    8 : "orange",
    9 :"gray"}
    
     
def _get_entropies_from_df(entropies_df, label):
    '''For some reason, each list of entropies were saved as a string.
    This method takes in a dataframe and returns all the entropies in that dataframe to
    np.ndarrays. '''
    return np.array(entropies_df[entropies_df["Label"] == label]["Persistence Entropies"].apply(
        lambda x: [float(text.strip()) for text in x[1:-2].split(",")]).tolist())


def main(): 
    entropies_df = pd.read_csv(PERS_ETP_OUTPUT_FILE, index_col=None)
    for recon in RECON_ROOT_NAMES:
        recon_df = entropies_df[entropies_df["Reconstruction"] == recon]
        
        incorrect_attack__df = recon_df[recon_df["Attack"] != "None"]
        cor_reg_entropies_df = recon_df[recon_df["Attack"] == "None"]

        inc_att_img_df= incorrect_attack__df[incorrect_attack__df["Input"] == "Image"]
        inc_att_cnn_df = incorrect_attack__df[incorrect_attack__df["Input"] == "CNN Output"]
        cor_reg_img_df = cor_reg_entropies_df[cor_reg_entropies_df["Input"] == "Image"]
        cor_reg_cnn_df = cor_reg_entropies_df[cor_reg_entropies_df["Input"] == "CNN Output"]

        fig, axes = plt.subplots(nrows=1, ncols=2, figsize = (20,10))

        for label in entropies_df["Label"].unique():

            inc_att_img_entrs = _get_entropies_from_df(inc_att_img_df, label)
            inc_att_cnn_entrs = _get_entropies_from_df(inc_att_cnn_df, label)
            cor_reg_img_entrs = _get_entropies_from_df(cor_reg_img_df, label)
            cor_reg_cnn_entrs = _get_entropies_from_df(cor_reg_cnn_df, label)

            axes[0].scatter(inc_att_cnn_entrs[:,0], 
                            inc_att_cnn_entrs[:,1],
                            c = COLOR_DICT[label],
                            label = label,
                            linewidths=5)



            axes[1].scatter(cor_reg_cnn_entrs[:,0],
                            cor_reg_cnn_entrs[:,1],
                            c = COLOR_DICT[label],
                            label = label,
                            linewidths=5)

        axes[0].set_xlabel("Input H0 entropes for Attack")
        axes[0].set_ylabel("Input H1 entropies for attack")
        axes[0].set_xlim([-4,4])
        axes[0].set_ylim([-4,4])
        axes[0].set_title(f"Entropies of Misclassified Attacked CNN Ouput on {recon}")
        axes[0].legend()

        axes[1].set_xlabel("Input H0 entropes")
        axes[1].set_ylabel("Input H1 entropies")
        axes[1].set_xlim([-4,4])
        axes[1].set_ylim([-4,4])
        axes[1].set_title(f"Entropies of Classified CNN Output on {recon}")
        axes[1].legend()
        fig.savefig(os.path.join(VISUAL_DIR, f"entrs_{recon}_cnn.png"))
        plt.close("all")


        
if __name__ == "__main__":
    main()