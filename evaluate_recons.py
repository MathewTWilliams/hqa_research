# Author: Matt Williams
# Version: 12/26/2022

import numpy as np
from utils import *
from torch.utils.data import DataLoader, Subset
import os
import torch
import torchattacks
from tqdm import tqdm
from numeric_image_folder import NumericImageFolder
from slice_torch_dataset import CombinedDataSet
from pytorch_cnn_base import query_model, outputs_to_predictions
from tsne import run_tsne
from sklearn.model_selection import train_test_split
from persistent_homology import *


def evaluate_dataset(model_name, test_labels, predictions, ds_name, recon_name, save_result = True, attack = None):
    """
    Arguments: 
    - model_name : name of the model that gave the predictions.
    - test_labels : list ground truth labels of the images. 
    - predictions : list of predicted labels made by the model. 
    - ds_name : name of the dataset that is being evaulated.
    - recon_name : the name of the reconstruction layer. 
    - ret_correct_idxs : should the method return the indicies of the images predicted correctly?
    - save_result: should this evaluation be saved to the classification_accuracies.csv file?
    - attack: the name of the attack that has been applied to the model (if any). """
    correct_idxs = []
    incorrect_idxs = []
    
    for i, (label, pred) in enumerate(zip(predictions, test_labels)):
        if label == pred:
            correct_idxs.append(i)
        elif label != pred:
            incorrect_idxs.append(i)

    if save_result:
        avg_accuracy = len(correct_idxs) / len(test_labels)
        attack_name = attack if attack != None else "None"
        #add_accuracy_results(model_name, ds_name, recon_name, attack_name, avg_accuracy)

    return correct_idxs, incorrect_idxs

def make_persistence_metrics(model, ds_test, predictions, ds_idxs, model_name, ds_name, root, attack):
    #Calculate Entropy of incorrectly classified points and attacked counterparts
    for idx in ds_idxs:
        img, label = ds_test[idx]
        pred = predictions[idx]
        atk_img = attack(img.unsqueeze(0), torch.LongTensor([label])).squeeze(0).detach().cpu()

        try:
            org_entr = calculate_entropy(img.numpy(), label, pred, root)
            atk_entr = calculate_entropy(atk_img.numpy(), label, pred, root, attack.attack)

            add_persistence_entropy(model_name, ds_name, label, pred, root, "None", "Image", org_entr)
            add_persistence_entropy(model_name, ds_name, label, pred, root, attack.attack, "Image", atk_entr)

        except IndexError as e: 
            print("IndexError: Persistence Inteval values were all infinity")

        try: 
            org_entr = calc_entropy_model_CNN_stack(model, img, label, pred, root)
            atk_entr = calc_entropy_model_CNN_stack(model, atk_img, label, pred, root, attack.attack)

            add_persistence_entropy(model_name, ds_name, label, pred, root, "None", "CNN Output", org_entr)
            add_persistence_entropy(model_name, ds_name, label, pred, root, attack.attack, "CNN Output", atk_entr)
            
        except IndexError as e:
            print("IndexError: Persistence Inteval values were all infinity")

        '''try:     
            org_img_pipeline = make_vectorized_persistence(img.numpy(), label, root)
            ak_img_pipeline = make_vectorized_persistence(atk_img.numpy(), label, root, attack.attack)

            add_vectorized_persistence(model_name, ds_name, label, pred, root, "None", org_img_pipeline)
            add_vectorized_persistence(model_name, ds_name, label, pred, root, attack.attack, ak_img_pipeline)

        except ValueError as e: 
            print("ValueError: Division by zero error in Scalar step on regular image")'''
        

def eval_model(model_save_path, model_name, dataset, root, num_classes):
    """
    Arguments: 
    - model_save_path: The file path where the model can be located. 
    - model_name: The name of the model to be evaluated.
    - dataset: file path to the dataset directory to use. 
    - root: which reconstruction layer is being evaluated in the given dataset.
    - num_classes: how many classes are in the dataset."""

    # Get Data set name
    ds_name = dataset.split("\\")[-1]

    #Load lenet model and make dataset & loader
    lenet_model = torch.load(model_save_path)
    lenet_model.eval()
    ds_test = NumericImageFolder(os.path.join(dataset, root), transform=IMG_FOLDER_TRANFORM)
    dl_test = DataLoader(ds_test, batch_size=MNIST_BATCH_SIZE, shuffle = False, num_workers = NUM_DATA_LOADER_WORKERS)
    
    #query model and evaluate results as normal
    model_output = query_model(lenet_model, dl_test, return_softmax = False)
    org_predictions = outputs_to_predictions(torch.Tensor(model_output))
    org_correct_idxs, org_incorrect_idxs = evaluate_dataset(model_name, ds_test.targets, org_predictions, ds_name, root, save_result = True)

    #make fgsm attack, query attacked model, evaluate the results
    fgsm_attack = torchattacks.FGSM(lenet_model)
    atk_model_output = query_model(lenet_model, dl_test, fgsm_attack, return_softmax = False)
    atk_predictions = outputs_to_predictions(torch.Tensor(atk_model_output))
    atk_correct_idxs, atk_incorrect_idxs = evaluate_dataset(model_name, ds_test.targets, atk_predictions, ds_name, root, True, fgsm_attack.attack)

    #Two examples of persistent barcode with misclassified point and its attacked counterpart.
    # One on the image and one on the output the CNN layers
    img, label = ds_test[atk_incorrect_idxs[0]]
    make_persistence_barcode(img.numpy(), label, root, False)
    barcode_model_CNN_Stack(lenet_model, img, label, root, False)
    atk_img = fgsm_attack(img.unsqueeze(0), torch.LongTensor([label])).squeeze(0).detach().cpu()
    make_persistence_barcode(atk_img.numpy(), label, root, True)
    barcode_model_CNN_Stack(lenet_model, atk_img, label, root, True)

    #Calculate Entropies
    #make_persistence_metrics(lenet_model, ds_test, org_predictions, org_correct_idxs, model_name, ds_name, root, fgsm_attack)
    make_persistence_metrics(lenet_model, ds_test, org_predictions, org_incorrect_idxs, model_name, ds_name, root, fgsm_attack)
    make_persistence_metrics(lenet_model, ds_test, atk_predictions, atk_correct_idxs, model_name, ds_name, root, fgsm_attack)
    #make_persistence_metrics(lenet_model, ds_test, atk_predictions, atk_incorrect_idxs, model_name, ds_name, root, fgsm_attack)
    
    '''#TSNE related
    if root in ["data_original", "data_recon_4"]:
        atk_mis_outputs = [atk_model_output[idx] for idx in atk_incorrect_idxs]
        atk_mis_true_labels = [ds_test.targets[idx] for idx in atk_incorrect_idxs]
        run_tsne(model_name, atk_mis_outputs, atk_mis_true_labels, ds_name, root, num_classes, "model output", fgsm_attack.attack, misclassified=True)

        test_idxs, _ = train_test_split(
            range(len(ds_test)),
            stratify = ds_test.targets,
            train_size = len(atk_incorrect_idxs),
            random_state = RANDOM_SEED
        )

        ds_test = Subset(ds_test, test_idxs)
        dl_test = DataLoader(ds_test, batch_size = MNIST_BATCH_SIZE, shuffle = False, num_workers = NUM_DATA_LOADER_WORKERS)
        
        new_model_outputs = query_model(lenet_model, dl_test, return_softmax = False)
        new_predictions = outputs_to_predictions(torch.Tensor(new_model_outputs))
        current_targets = [ds_test.dataset.targets[i] for i in test_idxs]
        correct_idxs,_ = evaluate_dataset(model_name, current_targets, new_predictions, ds_name, root, save_result = False)

        correct_outputs = [new_model_outputs[idx] for idx in correct_idxs]
        correct_targets = [current_targets[idx] for idx in correct_idxs]
        correct_images = [ds_test[idx][0].squeeze(0).numpy().flatten() for idx in correct_idxs]
        run_tsne(model_name, correct_outputs, correct_targets, ds_name, root, num_classes, "model output")
        run_tsne(model_name, correct_images, correct_targets, ds_name, root, num_classes, "input image")'''

def eval_tiled_model(model_save_path, model_name, dataset, root, num_classes, add_root = None):
    """
    Arguments: 
    - model_save_path: The file path where the model is be located. 
    - model_name: The name of the model to be evaluated.
    - dataset: file path to the dataset directory to use. 
    - root: which reconstruction layer is being evaluated in the given dataset.
    - num_classes: how many classes are in the dataset.
    - add_root: another optional reconstruction layer to be evalued. Used when we want 
    to evaluate slices of different reconstruciton layers of the same dataset."""
    
    # Get Data set name
    ds_name = dataset.split("\\")[-1]

    # Load lenet model
    lenet_model = torch.load(model_save_path)
    lenet_model.eval()

    # make datset & loader
    ds_test = NumericImageFolder(os.path.join(dataset, root), transform=IMG_FOLDER_TRANFORM)
    ds_test_2 = None
    root_name = root
    if add_root is not None: 
        ds_test_2 = NumericImageFolder(os.path.join(dataset, add_root), transform=IMG_FOLDER_TRANFORM)
        root_name += f"&{add_root.split('_')[-1]}"
    ds_test = CombinedDataSet(ds_test, ds_test_2, num_tiles=2, tile_split="v")
    dl_test = DataLoader(ds_test, batch_size=MNIST_BATCH_SIZE, shuffle = False, num_workers = NUM_DATA_LOADER_WORKERS)
        
    # query model and evaluate the results    
    model_output = query_model(lenet_model, dl_test, return_softmax = False)
    org_predictions = outputs_to_predictions(torch.Tensor(model_output))
    org_correct_idxs, org_incorrect_idxs = evaluate_dataset(model_name, ds_test._targets, org_predictions, ds_name, root_name, save_result=True)

    # make fgsm attack, query attacked model, evaluate the results
    fgsm_attack = torchattacks.FGSM(lenet_model)
    atk_model_output = query_model(lenet_model, dl_test, fgsm_attack, return_softmax = False)
    atk_predictions = outputs_to_predictions(torch.Tensor(atk_model_output))
    atk_correct_idxs, atk_incorrect_idxs = evaluate_dataset(model_name, ds_test._targets, atk_predictions, ds_name, root_name, True, fgsm_attack.attack)

    #Two examples of persistent barcode with misclassified point and its attacked counterpart.
    # One on the image and one on the output the CNN layers
    img, label = ds_test[atk_incorrect_idxs[0]]
    make_persistence_barcode(img.numpy(), label, f"Tiled_{root_name}", False)
    barcode_model_CNN_Stack(lenet_model, img, label, f"Tiled_{root_name}", False)
    atk_img = fgsm_attack(img.unsqueeze(0), torch.LongTensor([label])).squeeze(0).detach().cpu()
    make_persistence_barcode(atk_img.numpy(), label, f"Tiled_{root_name}", True)
    barcode_model_CNN_Stack(lenet_model, atk_img, label, f"Tiled_{root_name}", True)

    #Calculate Entropies
    make_persistence_metrics(lenet_model, ds_test, org_predictions, org_correct_idxs, model_name, ds_name, root_name, fgsm_attack)
    make_persistence_metrics(lenet_model, ds_test, org_predictions, org_incorrect_idxs, model_name, ds_name, root_name, fgsm_attack)
    make_persistence_metrics(lenet_model, ds_test, atk_predictions, atk_correct_idxs, model_name, ds_name, root_name, fgsm_attack)
    make_persistence_metrics(lenet_model, ds_test, atk_predictions, atk_incorrect_idxs, model_name, ds_name, root_name, fgsm_attack)

    '''if add_root is not None or root in ["data_original", "data_recon_3"]:
        atk_mis_outputs = [atk_model_output[idx] for idx in atk_incorrect_idxs]
        mis_true_labels = [ds_test._targets[idx] for idx in atk_incorrect_idxs]
        run_tsne(model_name, atk_mis_outputs, mis_true_labels, ds_name, root_name, num_classes, "model output", fgsm_attack.attack, misclassified=True)

        test_idxs, _ = train_test_split(
            range(len(ds_test)),
            stratify = ds_test._targets,
            train_size = len(atk_incorrect_idxs), 
            random_state = RANDOM_SEED
        )

        ds_test = Subset(ds_test, test_idxs)
        dl_test = DataLoader(ds_test, batch_size = MNIST_BATCH_SIZE, shuffle = False, num_workers=NUM_DATA_LOADER_WORKERS)
        
        new_model_outputs = query_model(lenet_model, dl_test, return_softmax = False)
        new_predictions = outputs_to_predictions(torch.Tensor(new_model_outputs))
        current_targets = [ds_test.dataset._targets[idx] for idx in test_idxs]
        correct_idxs,_ = evaluate_dataset(model_name, current_targets, new_predictions, ds_name, root_name, save_result = False)

        correct_outputs = [new_model_outputs[idx] for idx in correct_idxs]
        correct_targets = [current_targets[idx] for idx in correct_idxs]
        correct_images = [ds_test[idx][0].squeeze(0).numpy().flatten() for idx in correct_idxs]
        run_tsne(model_name, correct_outputs, correct_targets, ds_name, root_name, num_classes, "model output")
        run_tsne(model_name, correct_images, correct_targets, ds_name, root_name, num_classes, "input image")'''

def main():

    for root in RECON_ROOT_NAMES[4:]:

        eval_model(LENET_MNIST_PATH, "Lenet", IMG_MNIST_DIR_PATH, root, 10)
        #eval_model(LENET_FASH_MNIST_PATH, "Lenet", IMG_FASH_MNIST_DIR_PATH, root, 10)
        #eval_model(LENET_EMNIST_PATH, "Lenet", IMG_EMNIST_DIR_PATH, root, 47)
        
        if root != "data_recon_4":
            #eval_model(LENET_MNIST_PATH, "Lenet", IMG_MNIST_FFT_DIR_PATH, root, 10)
            
            eval_tiled_model(LENET_MNIST_PATH, "Lenet", IMG_TILED_MNIST_DIR_PATH, root,  10)
            #eval_tiled_model(LENET_FASH_MNIST_PATH, "Lenet", IMG_TILED_FASH_MNIST_DIR_PATH, root, 10)
            #eval_tiled_model(LENET_EMNIST_PATH, "Lenet", IMG_TILED_EMNIST_DIR_PATH, root, 47)

    eval_tiled_model(LENET_MNIST_PATH, "Lenet", IMG_TILED_MNIST_DIR_PATH, "data_recon_0", 10, "data_recon_3")

if __name__ == "__main__": 
    main()