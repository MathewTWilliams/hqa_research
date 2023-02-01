# Authors: Dr.Silvija Kokalj-Filipovic, Matt Williams
# Version: 1/10/2023

from gtda.images import Binarizer, RadialFiltration
from gtda.homology import CubicalPersistence
from gtda.diagrams import Scaler
from sklearn.pipeline import Pipeline
from gtda.diagrams import Amplitude
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import gudhi as gd
from gudhi.representations import Entropy
import numpy as np
from utils import VISUAL_DIR, add_vectorized_persistence, VECT_PERS_OUTPUT_FILE, device, \
add_persistence_entropy, PERS_ETP_OUTPUT_FILE, RECON_ROOT_NAMES
import os
import pandas as pd
from PIL import Image, ImageOps

def _show_img_tens(tensor, fname):
    data = tensor.detach().cpu().numpy()
    remmax = lambda x : x / x.max()
    remmin = lambda x : x - np.amin(x, axis = (0,1), keepdims = True)

    toint8 = lambda x : (remmax(remmin(x)) * (256 - 1e-4)).astype(np.uint8)
    img = Image.fromarray(toint8(data.squeeze(0))).convert('L')
    #img.thumbnail((512, 512))
    img = ImageOps.fit(img, (512, 512), method = 0, 
                        bleed = 0.0, centering = (0.5, 0.5))
    #img.show()
    img.save(fname)


def _diag_tidy(diag, eps = 1e-3):
    new_diag = []
    for _,x in diag:
        if np.abs(x[0] - x[1]) > eps:
            new_diag.append((_, x))

    return new_diag


def _get_CNN_stack_output(model, tensor_img):

    img_dim = [tensor_img.shape[-1], tensor_img.shape[-1]]
    dlmap = model._cnn_layers(tensor_img.unsqueeze(0).to(device))
    img_dim = [10, int(np.prod(dlmap.shape) / (dlmap.shape[0] * 10))]
    dm = dlmap.squeeze(2).squeeze(2)
    dlmap = (dm - dm.min(1, True)[0]) / (dm.max(1, True)[0] - dm.min(1, True)[0])
    dlmap = dlmap.view(dlmap.shape[0], 1, img_dim[0], img_dim[1])
    return dlmap.detach().cpu().squeeze(0)

def barcode_model_CNN_Stack(model, tensor_img, label, avatar, attacked = False):

    dlmap = _get_CNN_stack_output(model, tensor_img)
    _show_img_tens(dlmap, os.path.join(VISUAL_DIR, f"{label}_{avatar}_CNN.png"))
    #scaler
    #scaler = MinMaxScaler()
    #fit and scale in one step
    #normalized = scaler.fit_transform(dlmap.detach().cpu().numpy())
    new_avatar = f"{avatar}CNN"
    make_persistence_barcode(dlmap.numpy(), label, new_avatar, attacked)

def make_persistence_barcode(np_img, label, avatar = "", attacked = False):

    cc = gd.CubicalComplex(dimensions = (np_img.shape[1], np_img.shape[2]), 
            top_dimensional_cells = 1 - np_img.flatten())

    diag = cc.persistence()
    #fig = plt.figure(figsize=(3,3))
    diag_clean = _diag_tidy(diag, 1e-1)
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
    plt.close("all")

def calc_entropy_model_CNN_stack(model, tensor_img, true_label, pred_label, avatar, attack_name = "None"):
    dlmap = _get_CNN_stack_output(model, tensor_img)
    return calculate_entropy(dlmap.numpy(), true_label, pred_label, avatar, attack_name)

def calculate_entropy(np_img, true_label, pred_label, avatar, attack_name = "None"):

    remove_infinity = lambda barcode: np.array([bars for bars in barcode if bars[1] != np.inf])

    cc = gd.CubicalComplex(dimensions = (np_img.shape[1], np_img.shape[2]),
            top_dimensional_cells = 1 - np_img.flatten())

    _ = cc.persistence()
    inst0 = cc.persistence_intervals_in_dimension(0)
    inst1 = cc.persistence_intervals_in_dimension(1)
    # apply this operator to all barcodes
    inst0_normal = remove_infinity(inst0)
    inst1_normal = remove_infinity(inst1)
    PE = Entropy()
    inst0_ent = PE.fit_transform([inst0_normal]).flatten()[0]
    inst1_ent = PE.fit_transform([inst1_normal]).flatten()[0]
    entr = [inst0_ent, inst1_ent]
    print_mid = f" attacked" if attack_name == "None" else ""
    print(f"Entropies of {avatar} persistence bars for{print_mid} {true_label} declared {pred_label}: {inst0_ent} and {inst1_ent}")
    return entr

def make_vectorized_persistence(np_img, label, reconstruction, attack_name = "None"):
    
    steps = [
        ("binarizer", Binarizer(threshold=0.4)),
        ("filtration", RadialFiltration(center = np.array([20,6]))),
        ("diagram", CubicalPersistence()),
        ("rescaling", Scaler()),
        ("amplitude", Amplitude(metric="heat", metric_params={'sigma':0.15, 'n_bins':60}))
    ]

    heat_pipeline = Pipeline(steps)

    img_pipeline = heat_pipeline.fit_transform(np_img).flatten()
    print_end = f"{label}" if attack_name == "None" else f"attacked {label}"
    to_print = f"vectorized {reconstruction} persistence diagram {img_pipeline} for {print_end}"
    print(to_print)
    return img_pipeline

def _count_vectorized_persistence():
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


def _make_entropy_scatter_plots(ds_name, recon_layer, truth_label):

    pers_ent_df = pd.read_csv(PERS_ETP_OUTPUT_FILE, index_col=False)

    for recon in RECON_ROOT_NAMES: 
        cur_df = pers_ent_df[pers_ent_df["Dataset"] == ds_name &
                            pers_ent_df["Reconstruction"] == recon]

        entropies = cur_df["Persistence Entropies"].to_numpy()

        fig, axes = plt.subplots(nrows = 1, ncols = 2, figsize = (20,10))
        axes[0].scatter(entropies[:, 0])

    


if __name__ == "__main__":
    #_count_vectorized_persistence()
    _make_entropy_scatter_plots