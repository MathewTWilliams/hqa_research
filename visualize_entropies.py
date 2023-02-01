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

def main(): 
    entropies_df = pd.read_csv(PERS_ETP_OUTPUT_FILE, index_col=None)
    for recon in RECON_ROOT_NAMES:
        recon_entropies_df = entropies_df[entropies_df["Reconstruction"] == recon]
        
        inc_att_entropies_df = recon_entropies_df[recon_entropies_df["Attack"] != "None"]
        cor_reg_entropies_df = recon_entropies_df[recon_entropies_df["Attack"] == "None"]

        inc_att_img_entropies= inc_att_entropies_df[inc_att_entropies_df["Input"] == "Image"]
        inc_att_cnn_entropies = inc_att_entropies_df[inc_att_entropies_df["Input"] == "CNN Output"]
        cor_reg_img_entropies = cor_reg_entropies_df[cor_reg_entropies_df["Input"] == "Image"]
        cor_reg_cnn_entropies = cor_reg_entropies_df[cor_reg_entropies_df["Input"] == "CNN Output"]

        str_entr = inc_att_img_entropies["Persistence Entropies"].apply(
            lambda x: [float(text) for text in x[1:-2].split(",")])

        print(type(str_entr.tolist()[0][0]))


        '''fig, axes = plt.subplots(nrows=1, ncols=2, figsize = (20,10))
        axes[0].scatter(inc_att_img_entropies["Persistence Entropies"].to_numpy(), 
                        inc_att_img_entropies["Persistence Entropies"].to_numpy(),
                        c = inc_att_img_entropies["Label"], 
                        cmap="spring", linewidths=5)
        axes[0].set_xlabel("Input H0 entropes for Attack")
        axes[0].set_ylabel("Input H1 entropies for attack")
        axes[0].set_xlim([0,4])
        axes[0].set_ylim([0,4])



        axes[1].scatter(cor_reg_img_entropies["Perstence Entropies"].to_numpy(),
                        cor_reg_img_entropies["Persistence Entropies"].to_numpy(),
                        c = cor_reg_img_entropies["Label"])
        axes[1].set_xlabel("Input H0 entropes")
        axes[1].set_ylabel("Input H1 entropies")
        axes[1].set_xlim([0,4])
        axes[1].set_ylim([0,4])

        fig.savefig(os.path.join(VISUAL_DIR, f"entrs_{recon}_imgs.png"))'''
        break        

        
if __name__ == "__main__":
    main()