''' Author: Matt Williams
Version: 2/26/2023

'''

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from utils import WASS_DIST_OUTPUT_FILE, RECON_ROOT_NAMES, WASS_DIST_VIS_DIR, IMG_MNIST_DIR_PATH
import os
import random
from PIL import Image
from persistent_homology import calculate_wasserstein_distance
import numpy as np

def make_result_distributions():

    if not os.path.exists(WASS_DIST_VIS_DIR):
        os.mkdir(WASS_DIST_VIS_DIR)

    wasserstein_df = pd.read_csv(WASS_DIST_OUTPUT_FILE, index_col=False)

    unique_labels = wasserstein_df["Label"].unique()

    for recon in RECON_ROOT_NAMES: 
        for label in unique_labels:

            cur_wasserstein_df = wasserstein_df[(wasserstein_df["Reconstruction"] == recon)]
            
            wass_img_df = cur_wasserstein_df[cur_wasserstein_df["Input"] == "Image"]

            # if attack == "None" and the label, prediction 1, and prediction 2 are all the same value, 
            # then that's the wasserstein distance of two separate images of the same class correctly classified
            cor_wass_org_img_df = wass_img_df[(wass_img_df["Attack"] == "None") &
                                              (wass_img_df["Label"] == label) &
                                              (wass_img_df["Prediction 1(org)"] == label) & 
                                              (wass_img_df["Prediction 2(atk)"] == label)]

            inc_wass_atk_img_df = wass_img_df[(wass_img_df["Attack"] != "None") & 
                                              (wass_img_df["Label"] != label)
                                              (wass_img_df["Prediction 2(atk)"] == label)]

           
            fig, axes = plt.subplots(1,2)
            fig.set_figwidth(10)
            fig.set_figheight(8)
            
            sns.histplot(ax=axes[0], data = cor_wass_org_img_df, x = "Wasserstein_Distance", kde=True, stat="density", bins=25)
            axes[0].set_title(f"Wasserstein distances of classified {label} images produced by {recon}: {len(cor_wass_org_img_df.index)}", {"size": 9})
            axes[0].set_xlim(0,2.5)
            axes[0].set_ylim(0,0.4)

            sns.histplot(ax=axes[1], data = inc_wass_atk_img_df, x = "Wasserstein_Distance", kde=True, stat="density", bins=25)
            axes[1].set_title(f"Wasserstein distance of attacked images misclassified as {label} produced by {recon}: {len(inc_wass_atk_img_df.index)}", {"size": 9})
            axes[1].set_xlim(0,2.5)
            axes[1].set_ylim(0,0.4)

            fig.tight_layout()
            fig.savefig(os.path.join(WASS_DIST_VIS_DIR,f"{recon}_{label}.png"))
            plt.close("all")
            
if __name__ == "__main__":
    make_result_distributions()
