# Author: Matt Williams
# Version: 2/26/2023

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from utils import WASS_DIST_OUTPUT_FILE, RECON_ROOT_NAMES, WASS_DIST_VIS_DIR
import os

def main():
    if not os.path.exists(WASS_DIST_VIS_DIR):
        os.mkdir(WASS_DIST_VIS_DIR)

    wasserstein_df = pd.read_csv(WASS_DIST_OUTPUT_FILE, index_col=False)

    unique_labels = wasserstein_df["Label"].unique()

    for recon in RECON_ROOT_NAMES: 
        for label in unique_labels:

            cur_wasserstein_df = wasserstein_df[(wasserstein_df["Reconstruction"] == recon) &
                                                (wasserstein_df["Label"] == label)]
            
            wass_img_df = cur_wasserstein_df[cur_wasserstein_df["Input"] == "Image"]
            wass_cnn_df = cur_wasserstein_df[cur_wasserstein_df["Input"] == "CNN Output"]

            cor_wass_img_df = wass_img_df[wass_img_df["Prediction"] == label]
            inc_wass_img_df = wass_img_df[wass_img_df["Prediction"] != label]
            cor_wass_cnn_df = wass_cnn_df[wass_cnn_df["Prediction"] == label]
            inc_wass_cnn_df = wass_cnn_df[wass_cnn_df["Prediction"] != label]

            fig, axes = plt.subplots(2,2)
            fig.set_figwidth(15)
            fig.set_figheight(15)
            
            sns.histplot(ax=axes[0,0], data = cor_wass_img_df, x = "Wasserstein_Distance", kde=True, stat="probability", bins=25)
            axes[0,0].set_title(f"Wasserstein distances of classified {label} images produced by {recon}")
            axes[0,0].set_xlim(0,2.5)
            axes[0,0].set_ylim(0,0.5)

            sns.histplot(ax=axes[0,1], data = cor_wass_cnn_df, x = "Wasserstein_Distance", kde=True, stat="probability", bins=25)
            axes[0,1].set_title(f"Wasserstein distance of classified {label} CNN Outputs produced by {recon}")
            axes[0,1].set_xlim(0,2.5)
            axes[0,1].set_ylim(0,0.5)

            sns.histplot(ax=axes[1,0], data = inc_wass_img_df, x = "Wasserstein_Distance", kde=True, stat="probability", bins=25)
            axes[1,0].set_title(f"Wasserstein distance of misclassified {label} images produced by {recon}")
            axes[1,0].set_xlim(0,2.5)
            axes[1,0].set_ylim(0,0.5)

            sns.histplot(ax=axes[1,1], data = inc_wass_cnn_df, x = "Wasserstein_Distance", kde=True, stat="probability", bins=25)
            axes[1,1].set_title(f"Wasserstein distance of misclassified {label} CNN Outputs produced by {recon}")
            axes[1,1].set_xlim(0,2.5)
            axes[1,1].set_ylim(0, 0.5)
            
            fig.tight_layout()
            fig.savefig(os.path.join(WASS_DIST_VIS_DIR,f"{recon}_{label}.png"))
            plt.close("all")



            
if __name__ == "__main__":
    main()