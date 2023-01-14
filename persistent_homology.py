# Authors: Dr.Silvija Kokalj-Filipovic, Matt Williams
# Version: 1/10/2023

from gtda.images import Binarizer, RadialFiltration
from gtda.homology import CubicalPersistence
from gtda.diagrams import Scaler
from sklearn.pipeline import Pipeline
from gtda.diagrams import Amplitude
import matplotlib.pyplot as plt
import gudhi as gd
import numpy as np
from utils import VISUAL_DIR, add_vectorized_persistence, VECT_PERS_OUTPUT_FILE
import os
import pandas as pd

def diag_tidy(diag, eps = 1e-3):
    new_diag = []
    for _,x in diag:
        if np.abs(x[0] - x[1]) > eps:
            new_diag.append((_, x))

    return new_diag


def make_persistence_barcode(np_img, label, avatar = "", attacked = False):

    img_dim = np_img.shape[-1]
    cc = gd.CubicalComplex(dimensions = (img_dim, img_dim), 
            top_dimensional_cells = 1 - np_img.flatten())

    diag = cc.persistence()
    #fig = plt.figure(figsize=(3,3))
    diag_clean = diag_tidy(diag, 1e-3)
    gd.plot_persistence_barcode(diag_clean)

    plt.ylim(-1, len(diag_clean))
    plt.xticks(ticks=np.linspace(0,1,6), labels = np.round(np.linspace(1,0,6), 2))
    plt.yticks([])

    title_begin = "Attacked" if attacked else "Regular"
    title = f"{title_begin} {label} for {avatar}"
    plt.title(title)

    #plt.show()
    fig = plt.gcf()
    file_end = "_a.png" if attacked else ".png"
    file_name = f"Pers{label}_{avatar}{file_end}"
    fig.savefig(os.path.join(VISUAL_DIR, file_name))
    plt.clf()


def make_vectorized_persistence(np_img, label, model_name, ds_name, reconstruction, attack_name = "None"):
    
    steps = [
        ("binarizer", Binarizer(threshold=0.4)),
        ("filtration", RadialFiltration(center = np.array([20,6]))),
        ("diagram", CubicalPersistence()),
        ("rescaling", Scaler()),
        ("amplitude", Amplitude(metric="heat", metric_params={'sigma':0.15, 'n_bins':60}))
    ]

    heat_pipeline = Pipeline(steps)

    img_pipeline = heat_pipeline.fit_transform(np_img)
    add_vectorized_persistence(model_name, ds_name, label, reconstruction, attack_name, img_pipeline)
    print_end = f"{label}" if attack_name == "None" else f"attacked {label}"
    to_print = f"vectorized {reconstruction} persistence diagram {img_pipeline} for {print_end}"
    print(to_print)


def count_vectorized_persistence():
    vect_pers_df = pd.read_csv(VECT_PERS_OUTPUT_FILE, index_col=False)
    
    same_count = 0
    different_count = 0
    idx = 0

    while idx < len(vect_pers_df):
        cur_row_ds_1 = vect_pers_df.iloc[idx]
        cur_row_ds_2 = None
        if idx + 1 < len(vect_pers_df):
            cur_row_ds_2 = vect_pers_df.iloc[idx + 1]
        
        if cur_row_ds_2 is None:
            break

        cur_model_1 = cur_row_ds_1["Model"]
        cur_model_2 = cur_row_ds_2["Model"]
        cur_dataset_1 = cur_row_ds_1["Dataset"]
        cur_dataset_2 = cur_row_ds_2["Dataset"]
        cur_label_1 = cur_row_ds_1["Label"]
        cur_label_2 = cur_row_ds_2["Label"]
        cur_recon_1 = cur_row_ds_1["Reconstruction"]
        cur_recon_2 = cur_row_ds_2["Reconstruction"]
        cur_atk_1 = cur_row_ds_1["Attack"]
        cur_atk_2 = cur_row_ds_2["Attack"]

        if cur_model_1 != cur_model_2 or \
            cur_dataset_1 != cur_dataset_2 or \
            cur_label_1 != cur_label_2 or \
            cur_recon_1 != cur_recon_2 or \
            (cur_atk_1 != "None" and cur_atk_2 == "None"):

                idx += 1
                continue
        vp1 = cur_row_ds_1["Vectorized Persistence"]
        vp2 = cur_row_ds_2["Vectorized Persistence"]

        if vp1 == vp2: 
            same_count += 1
        else: 
            different_count += 1
        
        idx += 2

    print(f"Normal Image and Attacked Image have same Vectorized Persistence: {same_count}")
    print(f"Normal Image and Attacked Image have different Vectorized Persistence: {different_count}")


if __name__ == "__main__":
    count_vectorized_persistence()