# Author: Matt Williams
# Version: 12/26/2022

import numpy as np
from utils import *
from torch.utils.data import DataLoader, Subset
import os
import torch
from sklearn.metrics import confusion_matrix
import torchattacks
from tqdm import tqdm
from numeric_image_folder import NumericImageFolder
from slice_torch_dataset import CombinedDataSet
from pytorch_cnn_base import query_model, outputs_to_predictions
from tsne import run_tsne
from sklearn.model_selection import train_test_split

def evaluate_dataset(model_name, test_labels, predictions, ds_name, recon_name, attack = None):
    diag_counts = confusion_matrix(test_labels, predictions).diagonal().tolist()
    num_correct = np.sum(diag_counts)
    avg_accuracy = num_correct / len(test_labels)
    attack_name = attack if attack != None else "None"
    add_accuracy_results(model_name, ds_name, recon_name, attack_name, avg_accuracy)

    return len(test_labels) - num_correct


def eval_model(model_save_path, model_name, dataset, root, num_classes):
    ds_name = dataset.split("\\")[-1]

    lenet_model = torch.load(model_save_path)
    lenet_model.eval()
    ds_test = NumericImageFolder(os.path.join(dataset, root), transform=IMG_FOLDER_TRANFORM)
    dl_test = DataLoader(ds_test, batch_size=MNIST_BATCH_SIZE, shuffle = False, num_workers = NUM_DATA_LOADER_WORKERS)
        
    model_output = query_model(lenet_model, model_name, dl_test, ds_name, return_softmax = False, avatar=root)
    org_predictions = outputs_to_predictions(torch.Tensor(model_output))
    #_ = evaluate_dataset(model_name, ds_test.targets, org_predictions, ds_name, root)

    fgsm_attack = torchattacks.FGSM(lenet_model)
    atk_model_output = query_model(lenet_model, model_name, dl_test, ds_name, fgsm_attack, return_softmax = False, avatar=root)
    atk_predictions = outputs_to_predictions(torch.Tensor(atk_model_output))
    '''num_incorrect = evaluate_dataset(model_name, ds_test.targets, atk_predictions, ds_name, root, fgsm_attack.attack)

    if root in ["data_original", "data_recon_4"]:
        atk_output_labels = zip(atk_model_output, ds_test.targets)
        run_tsne(model_name, atk_output_labels, atk_predictions, ds_name, root, num_classes, fgsm_attack.attack)
    
        test_idxs, _ = train_test_split(
            range(len(ds_test)),
            stratify = ds_test.targets,
            train_size = num_incorrect,
            random_state = RANDOM_SEED
        )

        ds_test = Subset(ds_test, test_idxs)
        dl_test = DataLoader(ds_test, batch_size = MNIST_BATCH_SIZE, shuffle = False, num_workers = NUM_DATA_LOADER_WORKERS)
        new_model_outputs = query_model(lenet_model, model_name, dl_test, ds_name, return_softmax = False, avatar=root)
        new_predictions = outputs_to_predictions(torch.Tensor(new_model_outputs))
        targets = [ds_test.dataset.targets[i] for i in ds_test.indices]
        new_outputs_labels = zip(new_model_outputs, targets)
        run_tsne(model_name, new_outputs_labels, new_predictions, ds_name, root, num_classes, show_incorrect=False)'''


def eval_tiled_model(model_save_path, model_name, dataset, root, num_classes, add_root = None):
    ds_name = dataset.split("\\")[-1]

    lenet_model = torch.load(model_save_path)
    lenet_model.eval()

    ds_test = NumericImageFolder(os.path.join(dataset, root), transform=IMG_FOLDER_TRANFORM)
    ds_test_2 = None
    root_name = root
    if add_root is not None: 
        ds_test_2 = NumericImageFolder(os.path.join(dataset, add_root), transform=IMG_FOLDER_TRANFORM)
        root_name += f"&{add_root.split('_')[-1]}"
    ds_test = CombinedDataSet(ds_test, ds_test_2, num_tiles=2, tile_split="v")
    dl_test = DataLoader(ds_test, batch_size=MNIST_BATCH_SIZE, shuffle = False, num_workers = NUM_DATA_LOADER_WORKERS)
        
    model_output = query_model(lenet_model, model_name, dl_test, ds_name, return_softmax = False, avatar=root)
    org_predictions = outputs_to_predictions(torch.Tensor(model_output))
    _ = evaluate_dataset(model_name, ds_test._targets, org_predictions, ds_name, root_name)

    fgsm_attack = torchattacks.FGSM(lenet_model)
    atk_model_output = query_model(lenet_model, model_name, dl_test, ds_name, fgsm_attack, return_softmax = False, avatar=root)
    atk_predictions = outputs_to_predictions(torch.Tensor(atk_model_output))
    num_incorrect = evaluate_dataset(model_name, ds_test._targets, atk_predictions, ds_name, root_name, fgsm_attack.attack)

    if add_root is not None or root in ["data_original", "data_recon_3"]:
        atk_output_labels = zip(atk_model_output, ds_test._targets)
        run_tsne(model_name, atk_output_labels, atk_predictions, ds_name, root_name, num_classes, fgsm_attack.attack)

        test_idxs, _ = train_test_split(
            range(len(ds_test)),
            stratify = ds_test._targets,
            train_size = num_incorrect, 
            random_state = RANDOM_SEED
        )

        ds_test = Subset(ds_test, test_idxs)
        dl_test = DataLoader(ds_test, batch_size = MNIST_BATCH_SIZE, shuffle = False, num_workers=NUM_DATA_LOADER_WORKERS)
        new_model_outputs = query_model(lenet_model, model_name, dl_test, ds_name, return_softmax = False, avatar=root)
        new_predictions = outputs_to_predictions(torch.Tensor(new_model_outputs))
        targets = [ds_test.dataset._targets[i] for i in ds_test.indices]
        new_outputs_labels = zip(new_model_outputs, targets)
        run_tsne(model_name, new_outputs_labels, new_predictions, ds_name, root_name, num_classes, show_incorrect = False)


    
def main():

    for root in RECON_ROOT_NAMES:

        #eval_model(LENET_MNIST_PATH, "Lenet", IMG_MNIST_DIR_PATH, root, 10)
        #eval_model(LENET_FASH_MNIST_PATH, "Lenet", IMG_FASH_MNIST_DIR_PATH, root, 10)
        #eval_model(LENET_EMNIST_PATH, "Lenet", IMG_EMNIST_DIR_PATH, root, 47)

        eval_model(LENET_MNIST_PATH, "Lenet", IMG_MNIST_GELU_DIR_PATH, root, 10)

        '''if root != "data_recon_4":
            eval_model(LENET_MNIST_PATH, "Lenet", IMG_MNIST_FFT_DIR_PATH, root, 10)
            
            eval_tiled_model(LENET_MNIST_PATH, "Lenet", IMG_TILED_MNIST_DIR_PATH, root,  10)
            eval_tiled_model(LENET_FASH_MNIST_PATH, "Lenet", IMG_TILED_FASH_MNIST_DIR_PATH, root, 10)
            eval_tiled_model(LENET_EMNIST_PATH, "Lenet", IMG_TILED_EMNIST_DIR_PATH, root, 47)'''

    #eval_tiled_model(LENET_MNIST_PATH, "Lenet", IMG_TILED_MNIST_DIR_PATH, "data_recon_0", 10, "data_recon_3")

if __name__ == "__main__": 
    main()