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
def run_tsne(model_name, outputs_labels, predictions, ds_name, recon_name, num_classes, attack = None, show_incorrect = True):

    tsne_components = 2
    pca_components = 5

    points_to_show = []
    for (output, target), pred in zip(outputs_labels, predictions):
        if show_incorrect and target != pred:
            points_to_show.append((output, target))
        elif not show_incorrect and target == pred:
            points_to_show.append((output, target))

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
        title = f"{model_name}'s {points_shown} points on {ds_name} {ds_version} with {attack} Attack"
    plt.title(title)
    file_name = f"TSNE_{model_name}_{points_shown}_{ds_name}_{ds_version}.png"
    if attack is not None:
        file_name = f"TSNE_{model_name}_{points_shown}_{ds_name}_{ds_version}_{attack}.png"
    plt.legend(bbox_to_anchor=(1.01, 1), loc='upper left', borderaxespad=0)
    plt.tight_layout()
    plt.savefig(os.path.join(VISUAL_DIR, file_name))
    plt.clf()
