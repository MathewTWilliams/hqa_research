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
from gudhi.wasserstein import wasserstein_distance
from gudhi.representations import Entropy
import numpy as np
from utils import BARCODES_DIR, add_vectorized_persistence, VECT_PERS_OUTPUT_FILE, device, \
add_persistence_entropy, PERS_ETP_OUTPUT_FILE, RECON_ROOT_NAMES, MISC_VIS_DIR
import os
import pandas as pd
from PIL import Image, ImageOps

def _show_img_tens(tensor, fname):

    dir_name = os.path.dirname(fname)
    if not os.path.isdir(dir_name):
        os.makedirs(dir_name)
    

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
    _show_img_tens(dlmap, os.path.join(MISC_VIS_DIR, f"{label}_{avatar}_CNN.png"))
    #scaler
    #scaler = MinMaxScaler()
    #fit and scale in one step
    #normalized = scaler.fit_transform(dlmap.detach().cpu().numpy())
    new_avatar = f"{avatar}CNN"
    make_persistence_barcode(dlmap.numpy(), label, new_avatar, attacked)

def make_persistence_barcode(np_img, label, avatar = "", attacked = False):

    if not os.path.isdir(BARCODES_DIR):
        os.makedirs(BARCODES_DIR)

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
    fig.savefig(os.path.join(BARCODES_DIR, file_name))
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

def calc_wass_dist_CNN_stack(model, org_tensor_img, atk_tensor_img, true_label, org_pred, atk_pred, avatar, attack_name): 
    org_dlmap = _get_CNN_stack_output(model, org_tensor_img)
    atk_dlmap = _get_CNN_stack_output(model, atk_tensor_img)
    return calculate_wasserstein_distance(org_dlmap.numpy(), atk_dlmap.numpy(), true_label, org_pred, atk_pred, avatar, attack_name)


def calculate_wasserstein_distance(org_np_img, atk_np_img, true_label, org_pred, atk_pred,  avatar, attack_name = "FGSM"):

    org_cc = gd.CubicalComplex(dimensions = (org_np_img.shape[1], org_np_img.shape[2]), 
                            top_dimensional_cells = 1 - org_np_img.flatten())

    org_diag = org_cc.persistence()
    org_dgels = np.asarray([list(dl[1])for dl in org_diag])

    atk_cc = gd.CubicalComplex(dimensions = (atk_np_img.shape[1], atk_np_img.shape[2]), 
                            top_dimensional_cells = 1 - atk_np_img.flatten())

    atk_diag = atk_cc.persistence()
    atk_dgels = np.asarray([list(dl[1]) for dl in atk_diag])

    print("Caculating Distance Metric")
    wass_dist = wasserstein_distance(org_dgels, atk_dgels, matching = False, order = 1.0, internal_p = 2)
    print(f"Wasserstein distance of {avatar} persistence bars for {true_label} normal declared {org_pred} and attack declared {atk_pred}: {wass_dist}")

    return wass_dist

def make_vectorized_persistence(np_img, label, pred, reconstruction, attack_name = "None"):
    
    steps = [
        ("binarizer", Binarizer(threshold=0.4)),
        ("filtration", RadialFiltration(center = np.array([20,6]))),
        ("diagram", CubicalPersistence()),
        ("rescaling", Scaler()),
        ("amplitude", Amplitude(metric="heat", metric_params={'sigma':0.15, 'n_bins':60}))
    ]

    heat_pipeline = Pipeline(steps)

    img_pipeline = heat_pipeline.fit_transform(np_img).flatten()
    print_end = f"{label} predicted as {pred}" if attack_name == "None" else f"attacked {label} predicted as {pred}"
    to_print = f"vectorized {reconstruction} persistence diagram {img_pipeline} for {print_end}"
    print(to_print)
    return img_pipeline


# example of calling vectorized_persistence
'''try:     
    org_img_pipeline = make_vectorized_persistence(img.numpy(), label, root)
    add_vectorized_persistence(model_name, ds_name, label, org_pred, root, "None", org_img_pipeline)

    except ValueError as e: 
        print("ValueError: Division by zero error in Scalar step on regular image")

    try:     
        ak_img_pipeline = make_vectorized_persistence(atk_img.numpy(), label, root, attack.attack)
        add_vectorized_persistence(model_name, ds_name, label, atk_pred, root, attack.attack, ak_img_pipeline)

    except ValueError as e: 
        print("ValueError: Division by zero error in Scalar step on regular image")'''