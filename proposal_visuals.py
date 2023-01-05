# Author: Matt Williams
# Version: 1/3/2023

import torch
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import torchattacks
from numeric_image_folder import NumericImageFolder
from torch.utils.data import DataLoader, Subset
import numpy as np
import os
import seaborn as sns
from utils import *
import matplotlib.pyplot as plt
from pytorch_cnn_base import run_predictions
from sklearn.model_selection import train_test_split
from slice_torch_dataset import CombinedDataSet

def run_tsne(model_name, ds_test, predictions, ds_name, recon_name, num_classes, attack = None, show_incorrect = True):
    tsne_components = 2
    pca_components = 50

    points_to_show = []
    for (img, target), pred in zip(ds_test, predictions):
        if show_incorrect and target != pred:
            points_to_show.append((img.squeeze(0).numpy().flatten(), target))
        elif not show_incorrect and target == pred:
            points_to_show.append((img.squeeze(0).numpy().flatten(), target))

    train = [data[0] for data in points_to_show]
    labels = [data[1] for data in points_to_show]
    train = StandardScaler().fit_transform(train)
    pca = PCA(n_components=pca_components, random_state=42)
    pca_res = pca.fit_transform(train)
    tsne = TSNE(n_components=tsne_components, random_state = 42, init="pca", learning_rate="auto")
    tsne_res = tsne.fit_transform(pca_res)

    sns.scatterplot(x = tsne_res[:,0], y = tsne_res[:,1], hue = labels, palette = sns.hls_palette(num_classes), legend = "full")

    points_shown = "misclassified" if show_incorrect else "classified"
    ds_version = "_".join(recon_name.split("_")[1:])
    title = f"{model_name}'s {points_shown} points on {ds_name} {ds_version}"
    if attack is not None: 
        title = f"{model_name}'s {points_shown} points on {ds_name} {ds_version} with {attack.attack} Attack"
    plt.title(title)
    file_name = f"TSNE_{model_name}_{points_shown}_{ds_name}_{ds_version}.png"
    if attack is not None:
        file_name = f"TSNE_{model_name}_{points_shown}_{ds_name}_{ds_version}_{attack.attack}.png"
    plt.legend(bbox_to_anchor=(1.01, 1), loc='upper left', borderaxespad=0)
    plt.tight_layout()
    plt.savefig(os.path.join(VISUAL_DIR, file_name))
    plt.clf()

    num_incorrect = len(points_to_show) if show_incorrect else len(ds_test) - len(points_shown)
    return num_incorrect


def run_model(model_save_path, model_name, dataset, root, num_classes, run_attack = False, show_incorrect = True):
    lenet_model = torch.load(model_save_path)
    lenet_model.eval()
    
    ds_name = dataset.split("\\")[-1]

    attack = None
    if run_attack:
        attack = torchattacks.FGSM(lenet_model)

    ds_test = NumericImageFolder(os.path.join(dataset, root), transform=IMG_FOLDER_TRANFORM)
    dl_test = DataLoader(ds_test, batch_size=MNIST_BATCH_SIZE, shuffle = False, num_workers = NUM_DATA_LOADER_WORKERS)
    predictions = run_predictions(lenet_model, dl_test, attack)
    run_tsne(model_name, ds_test, predictions, ds_name, root, num_classes, attack, show_incorrect)
        

if __name__ == "__main__":
    roots = ["data_original", "data_recon_4"]

    # running through generic TSNE visualizations showing misclassified points with true values
    for root in roots:
        run_model(LENET_MNIST_PATH, "Lenet", IMG_MNIST_DIR_PATH, root, 10)
        run_model(LENET_FASH_MNIST_PATH, "Lenet", IMG_FASH_MNIST_DIR_PATH, root, 10)
        run_model(LENET_EMNIST_PATH, "Lenet", IMG_EMNIST_DIR_PATH, root, 47)

        run_model(LENET_MNIST_PATH, "Lenet", IMG_MNIST_DIR_PATH, root, 10, run_attack = True)
        run_model(LENET_FASH_MNIST_PATH, "Lenet", IMG_FASH_MNIST_DIR_PATH, root, 10, run_attack = True)
        run_model(LENET_EMNIST_PATH, "Lenet", IMG_EMNIST_DIR_PATH, root, 47, run_attack = True) 

    # more specific visualizations
    mnist_original_fgsm_num_incorrect = 374
    ds_test = NumericImageFolder(os.path.join(IMG_MNIST_DIR_PATH, "data_original"), transform = IMG_FOLDER_TRANFORM)
    test_idxs, _ = train_test_split(
        range(len(ds_test)),
        stratify=ds_test.targets,
        train_size = mnist_original_fgsm_num_incorrect,
        random_state = 42
    )

    ds_test = Subset(ds_test, test_idxs)
    dl_test = DataLoader(ds_test, batch_size=MNIST_BATCH_SIZE, shuffle = False, num_workers=NUM_DATA_LOADER_WORKERS)
    lenet_model = torch.load(LENET_MNIST_PATH)
    ds_name = "MNIST"
    predictions = run_predictions(lenet_model, dl_test)
    run_tsne("Lenet", ds_test, predictions, ds_name, "data_original", 10, None, show_incorrect=False)


    mnist_recon_4_fgsm_num_incorrect = 2200
    ds_test = NumericImageFolder(os.path.join(IMG_MNIST_DIR_PATH, "data_original"), transform = IMG_FOLDER_TRANFORM)
    test_idxs, _ = train_test_split(
        range(len(ds_test)),
        stratify=ds_test.targets,
        train_size = mnist_recon_4_fgsm_num_incorrect,
        random_state = 42
    )

    ds_test = Subset(ds_test, test_idxs)
    dl_test = DataLoader(ds_test, batch_size=MNIST_BATCH_SIZE, shuffle = False, num_workers=NUM_DATA_LOADER_WORKERS)
    lenet_model = torch.load(LENET_MNIST_PATH)
    ds_name = "MNIST"
    predictions = run_predictions(lenet_model, dl_test)
    run_tsne("Lenet", ds_test, predictions, ds_name, "data_recon_4", 10, None, show_incorrect=False)


    lenet_model = torch.load(LENET_MNIST_PATH)
    attack = torchattacks.FGSM(lenet_model)
    ds_test = NumericImageFolder(os.path.join(IMG_TILED_MNIST_DIR_PATH, "data_recon_3"), transform = IMG_FOLDER_TRANFORM)
    ds_test = CombinedDataSet(ds_test, num_tiles=2, tile_split="v")
    dl_test = DataLoader(ds_test, batch_size=MNIST_BATCH_SIZE, shuffle=False, num_workers=NUM_DATA_LOADER_WORKERS)
    predictions = run_predictions(lenet_model, dl_test, attack)
    num_incorrect = run_tsne("Lenet", ds_test, predictions, "Tiled_MNIST", "data_recon_3", 10, attack, show_incorrect = True)
    test_idxs, _ = train_test_split(
        range(len(ds_test)),
        stratify = ds_test._targets,
        train_size = num_incorrect,
        random_state = 42 
    )

    ds_test = Subset(ds_test, test_idxs)
    dl_test = DataLoader(ds_test, batch_size = MNIST_BATCH_SIZE, shuffle = False, num_workers = NUM_DATA_LOADER_WORKERS)
    predictions = run_predictions(lenet_model, dl_test)
    _ = run_tsne("Lenet", ds_test, predictions, "Tiled_MNIST", "data_recon_3", 10, None, show_incorrect = False)