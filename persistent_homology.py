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
from utils import VISUAL_DIR, add_vectorized_persistence
import os

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
    file_name = f"Pers{label}_{avatar}_{file_end}"
    fig.savefig(os.path.join(VISUAL_DIR, file_name))


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

def main():
    pass



if __name__ == "__main__":
    main()