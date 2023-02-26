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
            

            wass_reg_img_df = cur_wasserstein_df[cur_wasserstein_df["Input"] == "Image"]
            wass_reg_cnn_df = cur_wasserstein_df[cur_wasserstein_df["Input"] == "CNN Output"]
            wass_atk_img_df = cur_wasserstein_df[cur_wasserstein_df["Input"] == "Image"]
            wass_atk_cnn_df = cur_wasserstein_df[cur_wasserstein_df["Input"] == "CNN Output"]

            cor_wass_reg_img_df = wass_reg_img_df[wass_reg_img_df["Prediction"] == label]
            cor_wass_reg_cnn_df = wass_reg_cnn_df[wass_reg_cnn_df["Prediction"] == label]
            inc_wass_reg_img_df = wass_reg_img_df[wass_reg_img_df["Prediction"] != label]
            inc_wass_reg_cnn_df = wass_reg_cnn_df[wass_reg_cnn_df["Prediction"] != label]

            cor_wass_atk_img_df = wass_atk_img_df[wass_atk_img_df["Prediction"] == label]
            cor_wass_atk_cnn_df = wass_atk_cnn_df[wass_atk_cnn_df["Prediction"] == label]
            inc_wass_atk_img_df = wass_atk_img_df[wass_atk_img_df["Prediction"] != label]
            inc_wass_atk_cnn_df = wass_atk_cnn_df[wass_atk_cnn_df["Prediction"] != label]


            _ = sns.histplot(data = cor_wass_reg_img_df,
                                 x = "Wasserstein_Distance"
            )
            plt.title(f"Wasserstein distance of classified {label} images produced by {recon}")
            plt.savefig(os.path.join(WASS_DIST_VIS_DIR,f"{recon}_cor_reg_img_{label}.png"))
            plt.close("all")

            _ = sns.histplot(data = cor_wass_reg_cnn_df,
                                 x = "Wasserstein_Distance"
            )
            plt.title(f"Wasserstein distance of classified {label} CNN Outputs produced by {recon}")
            plt.savefig(os.path.join(WASS_DIST_VIS_DIR,f"{recon}_cor_reg_cnn_{label}.png"))
            plt.close("all")

            _ = sns.histplot(data = inc_wass_reg_img_df,
                                 x = "Wasserstein_Distance"
            )
            plt.title(f"Wasserstein distance of misclassified {label} images produced by {recon}")
            plt.savefig(os.path.join(WASS_DIST_VIS_DIR,f"{recon}_inc_reg_img_{label}.png"))
            plt.close("all")

            _ = sns.histplot(data = inc_wass_reg_cnn_df,
                                 x = "Wasserstein_Distance"
            )
            plt.title(f"Wasserstein distance of misclassified {label} CNN Outputs produced by {recon}")
            plt.savefig(os.path.join(WASS_DIST_VIS_DIR,f"{recon}_inc_reg_cnn_{label}.png"))
            plt.close("all")

            _ = sns.histplot(data = cor_wass_atk_img_df,
                                 x = "Wasserstein_Distance"
            )
            plt.title(f"Wasserstein distance of attacked classified {label} images produced by {recon}")
            plt.savefig(os.path.join(WASS_DIST_VIS_DIR,f"{recon}_cor_atk_img_{label}.png"))
            plt.close("all")

            _ = sns.histplot(data = cor_wass_atk_cnn_df,
                                 x = "Wasserstein_Distance"
            )
            plt.title(f"Wasserstein distance of attacked classified {label} CNN Outputs produced by {recon}")
            plt.savefig(os.path.join(WASS_DIST_VIS_DIR,f"{recon}_cor_atk_cnn_{label}.png"))
            plt.close("all")

            _ = sns.histplot(data = inc_wass_atk_img_df,
                                 x = "Wasserstein_Distance"
            )
            plt.title(f"Wasserstein distance of attacked misclassified {label} images produced by {recon}")
            plt.savefig(os.path.join(WASS_DIST_VIS_DIR,f"{recon}_inc_atk_img_{label}.png"))
            plt.close("all")

            _ = sns.histplot(data = inc_wass_atk_cnn_df,
                                 x = "Wasserstein_Distance"
            )
            plt.title(f"Wasserstein distance of attacked misclassified {label} CNN Outputs produced by {recon}")
            plt.savefig(os.path.join(WASS_DIST_VIS_DIR,f"{recon}_inc_atk_cnn_{label}.png"))
            plt.close("all")

            
if __name__ == "__main__":
    main()