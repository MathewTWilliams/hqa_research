# Author: Matt Williams
# Version: 1/3/2023

import torch
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import torchattacks
from numeric_image_folder import NumericImageFolder
from torch.utils.data import DataLoader
import numpy as np
import os
import seaborn as sns
from utils import *
import matplotlib.pyplot as plt


def run_tsne(model, model_name, dl_test, ds_name, recon_name, num_classes, attack = None):
    all_outputs = torch.Tensor().to(device)
    test_labels = []
    tsne_components = 2
    pca_components = 50

    for data, labels in dl_test:
        if attack != None:
            data = attack(data, labels)
        cur_output = model(data.to(device))
        all_outputs = torch.cat((all_outputs, cur_output),0)
        test_labels.extend(labels.tolist())


    softmax_probs = torch.exp(all_outputs).detach().cpu().numpy()
    predictions = np.argmax(softmax_probs, axis = -1)

    miss_classified = []
    for (img, target), pred in zip(dl_test.dataset, predictions):
        if target != pred:
            miss_classified.append((img.squeeze(0).numpy().flatten(), target, pred))

    train = [data[0] for data in miss_classified]
    labels = [data[1] for data in miss_classified]
    preds = [data[2] for data in miss_classified]
    train = StandardScaler().fit_transform(train)
    pca = PCA(n_components=pca_components, random_state=42)
    pca_res = pca.fit_transform(train)
    tsne = TSNE(n_components=tsne_components, random_state = 42)
    tsne_res = tsne.fit_transform(pca_res)

    sns.scatterplot(x = tsne_res[:,0], y = tsne_res[:,1], hue = labels, palette = sns.hls_palette(num_classes), legend = "full")
    ds_version = "_".join(recon_name.split("_")[1:])
    title = f"{model_name}'s misclassified points on {ds_name} {ds_version}"
    if attack is not None: 
        title = f"{model_name}'s misclassified points on {ds_name} {ds_version} with {attack.attack} Attack"
    plt.title(title)
    file_name = f"TSNE_{model_name}_{ds_name}_{ds_version}.png"
    if attack is not None:
        file_name = f"TSNE_{model_name}_{ds_name}_{ds_version}_{attack.attack}.png"
    plt.legend(bbox_to_anchor=(1.01, 1), loc='upper left', borderaxespad=0)
    plt.tight_layout()
    plt.savefig(os.path.join(VISUAL_DIR, file_name))
    plt.clf()



def run_model(model_save_path, model_name, dataset, num_classes):
    roots = ["data_original", "data_recon_4"]
    lenet_model = torch.load(model_save_path)
    lenet_model.eval()
    
    ds_name = dataset.split("\\")[-1]
    for root in roots:
        ds_test = NumericImageFolder(os.path.join(dataset, root), transform=IMG_FOLDER_TRANFORM)
        dl_test = DataLoader(ds_test, batch_size=MNIST_BATCH_SIZE, shuffle = False, num_workers = 4)
        run_tsne(lenet_model, model_name, dl_test, ds_name, root, num_classes)
        
    fgsm_attack = torchattacks.FGSM(lenet_model)

    for root in roots:
        ds_test = NumericImageFolder(os.path.join(dataset, root), transform=IMG_FOLDER_TRANFORM)
        dl_test = DataLoader(ds_test, batch_size=MNIST_BATCH_SIZE, shuffle=False, num_workers=4)
        run_tsne(lenet_model, model_name, dl_test, ds_name, root, num_classes, attack = fgsm_attack)


if __name__ == "__main__":
    run_model(LENET_MNIST_PATH, "Lenet", IMG_MNIST_DIR_PATH, 10)
    run_model(LENET_FASH_MNIST_PATH, "Lenet", IMG_FASH_MNIST_DIR_PATH, 10)
    run_model(LENET_EMNIST_PATH, "Lenet", IMG_EMNIST_DIR_PATH, 47)
