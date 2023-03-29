# Author: Matt Williams
# Version: 1/3/2023

import torch
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import os
import seaborn as sns
from utils import *
import matplotlib.pyplot as plt
import numpy as np
def run_tsne(model_name, data, labels, ds_name, recon_name, num_classes, data_desc, attack = None, misclassified = False):

    tsne_components = 2
    pca_components = 5

    if not os.path.isdir(TSNE_DIR):
        os.makedirs(TSNE_DIR)


    data = StandardScaler().fit_transform(data)
    pca = PCA(n_components=pca_components, random_state=42)
    pca_res = pca.fit_transform(data)
    tsne = TSNE(n_components=tsne_components, random_state = 42, init="pca", learning_rate="auto")
    tsne_res = tsne.fit_transform(pca_res)

    sns.scatterplot(x = tsne_res[:,0], y = tsne_res[:,1], hue = labels, palette = sns.hls_palette(num_classes), legend = "full")

    points_shown = "misclassified" if misclassified else "classified"
    ds_version = "_".join(recon_name.split("_")[1:])
    title = f"{model_name}'s {points_shown} {data_desc} on {ds_name} {ds_version}"
    if attack is not None: 
        title = f"{model_name}'s {points_shown} {data_desc} on {ds_name} {ds_version} with {attack} Attack"
    plt.title(title, {'size' : 10})
    data_desc = "_".join(data_desc.split(" "))
    file_name = f"TSNE_{model_name}_{points_shown}_{ds_name}_{ds_version}_{data_desc}.png"
    if attack is not None:
        file_name = f"TSNE_{model_name}_{points_shown}_{ds_name}_{ds_version}_{data_desc}_{attack}.png"
    plt.legend(bbox_to_anchor=(1.01, 1), loc='upper left', borderaxespad=0)
    plt.tight_layout()
    plt.savefig(os.path.join(TSNE_DIR, file_name))
    plt.clf()
