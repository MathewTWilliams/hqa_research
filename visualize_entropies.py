# Author: Matt Williams
# Version: 1/31/2023

import pandas as pd
import matplotlib.pyplot as plt
from utils import *
import os
import numpy as np
import seaborn as sns


def main(): 

    if not os.path.isdir(ENTR_VIS_DIR):
        os.makedirs(ENTR_VIS_DIR)

    entropies_df = pd.read_csv(PERS_ETP_OUTPUT_FILE, index_col=None)
    for recon in RECON_ROOT_NAMES:
        recon_df = entropies_df[entropies_df["Reconstruction"] == recon]

        h0_min = recon_df['H0'].min() - 0.25
        h0_max = recon_df['H0'].max() + 0.25
        h1_min = recon_df['H1'].min() - 0.25
        h1_max = recon_df['H1'].max() + 0.25
        
        recon_atk_df = recon_df[recon_df["Attack"] != "None"]
        recon_org_df = recon_df[recon_df["Attack"] == "None"]

        recon_atk_img_df = recon_atk_df[recon_atk_df["Input"] == "Image"]
        recon_atk_cnn_df = recon_atk_df[recon_atk_df["Input"] == "CNN Output"]
        recon_org_img_df = recon_org_df[recon_org_df["Input"] == "Image"]
        recon_org_cnn_df = recon_org_df[recon_org_df["Input"] == "CNN Output"]

        cor_recon_org_img_df = recon_org_img_df.query("Label == Prediction")
        inc_recon_org_img_df = recon_org_img_df.query("Label != Prediction")
        cor_recon_org_cnn_df = recon_org_cnn_df.query("Label == Prediction")
        inc_recon_org_cnn_df = recon_org_cnn_df.query("Label != Prediction")

        cor_recon_atk_img_df = recon_atk_img_df.query("Label == Prediction")
        inc_recon_atk_img_df = recon_atk_img_df.query("Label != Prediction")
        cor_recon_atk_cnn_df = recon_atk_cnn_df.query("Label == Prediction")
        inc_recon_atk_cnn_df = recon_atk_cnn_df.query("Label != Prediction")

        fig, axs = plt.subplots(nrows = 2, ncols = 4)
        fig.set_figwidth(20)
        fig.set_figheight(15)

        sns.boxplot(x = "Label", y = "H0", data = cor_recon_org_img_df, ax = axs[0,0])
        axs[0,0].set_title("Classified Normal Images")
        sns.boxplot(x = "Label", y = "H1", data = cor_recon_org_img_df, ax = axs[1,0])
        axs[1,0].set_title("Classified Normal Images")
        sns.boxplot(x = "Label", y = "H0", data = cor_recon_atk_img_df, ax = axs[0,1])
        axs[0,1].set_title("Classified Attacked Images")
        sns.boxplot(x = "Label", y = "H1", data = cor_recon_atk_img_df, ax = axs[1,1])
        axs[1,1].set_title("Classified Attacked Images")
        sns.boxplot(x = "Label", y = "H0", data = inc_recon_org_img_df, ax = axs[0,2])
        axs[0,2].set_title("Misclassified Normal Images")
        sns.boxplot(x = "Label", y = "H1", data = inc_recon_org_img_df, ax = axs[1,2])
        axs[1,2].set_title("Misclassified Normal Images")
        sns.boxplot(x = "Label", y = "H0", data = inc_recon_atk_img_df, ax = axs[0,3])
        axs[0,3].set_title("Misclassified Attacked Images")
        sns.boxplot(x = "Label", y = "H1", data = inc_recon_atk_img_df, ax = axs[1,3])
        axs[1,3].set_title("Misclassified Attacked Images")

        for ax in axs[0,:]:
            ax.set_ylim(h0_min, h0_max)
        for ax in axs[1,:]:
            ax.set_ylim(h1_min, h0_max)

        plt.tight_layout()
        fig.savefig(os.path.join(ENTR_VIS_DIR, f"{recon}_entropies_imgs.png"))
        plt.close("all")

        fig, axs = plt.subplots(nrows = 2, ncols=4)
        fig.set_figwidth(20)
        fig.set_figheight(15)

        sns.boxplot(x = "Label", y = "H0", data = cor_recon_org_cnn_df, ax = axs[0,0])
        axs[0,0].set_title("Classified Normal CNN Output")
        sns.boxplot(x = "Label", y = "H1", data = cor_recon_org_cnn_df, ax = axs[1,0])
        axs[1,0].set_title("Classified Normal CNN Output")
        sns.boxplot(x = "Label", y = "H0", data = cor_recon_atk_cnn_df, ax = axs[0,1])
        axs[0,1].set_title("Classified Attacked CNN Output")
        sns.boxplot(x = "Label", y = "H1", data = cor_recon_atk_cnn_df, ax = axs[1,1])
        axs[1,1].set_title("Classified Attacked CNN Output")
        sns.boxplot(x = "Label", y = "H0", data = inc_recon_org_cnn_df, ax = axs[0,2])
        axs[0,2].set_title("Misclassified Normal CNN Output")
        sns.boxplot(x = "Label", y = "H1", data = inc_recon_org_cnn_df, ax = axs[1,2])
        axs[1,2].set_title("Misclassified Normal CNN Output")
        sns.boxplot(x = "Label", y = "H0", data = inc_recon_atk_cnn_df, ax = axs[0,3])
        axs[0,3].set_title("Misclassified Attacked CNN Output")
        sns.boxplot(x = "Label", y = "H1", data = inc_recon_atk_cnn_df, ax = axs[1,3])
        axs[1,3].set_title("Misclassified Attacked CNN Output")

        for ax in axs[0,:]:
            ax.set_ylim(h0_min, h0_max)
        for ax in axs[1,:]:
            ax.set_ylim(h1_min, h0_max)

        plt.tight_layout()
        fig.savefig(os.path.join(ENTR_VIS_DIR, f"{recon}_entropies_cnns.png"))
        plt.close("all")
        


if __name__ == "__main__":
    main()