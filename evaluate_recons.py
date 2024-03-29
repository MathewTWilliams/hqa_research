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
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import random
 
from reconstruct_avg_attack import run_reconstruct_avg, run_dict_reconstruct_avg

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
        if not os.path.isdir(CONF_MAT_VIS_DIR): 
            os.mkdir(CONF_MAT_VIS_DIR)
        conf_mat_disp = ConfusionMatrixDisplay.from_predictions(test_labels, predictions, normalize = "all")
        conf_mat_disp = conf_mat_disp.plot()
        plt.title(f"{model_name} Confusion Matrix on {ds_name}: {recon_name}")
        plt.savefig(os.path.join(CONF_MAT_VIS_DIR, f"{model_name}_{ds_name}_{recon_name}.png"))
        plt.close("all")

        avg_accuracy = len(correct_idxs) / len(test_labels)
        attack_name = attack if attack != None else "None"
        add_accuracy_results(model_name, ds_name, recon_name, attack_name, avg_accuracy)

    return correct_idxs, incorrect_idxs

def make_persistence_metrics(model, ds_test, org_predictions, atk_predictions, model_name, ds_name, root, attack):
    #Calculate Entropy of incorrectly classified points and attacked counterparts

    for idx, (img, label) in enumerate(ds_test):
        atk_img = attack(img.unsqueeze(0), torch.LongTensor([label])).squeeze(0).detach().cpu()
        org_pred = org_predictions[idx]
        atk_pred = atk_predictions[idx]
        
        try:
            org_entr = calculate_entropy(img.numpy(), label, org_pred, root)
            add_persistence_entropy(model_name, ds_name, label, org_pred, root, "None", "Image", org_entr)

        except IndexError as e: 
            print("IndexError: Persistence Entropy values were all infinity")

        try:
            atk_entr = calculate_entropy(atk_img.numpy(), label, atk_pred, root, attack.attack)
            add_persistence_entropy(model_name, ds_name, label, atk_pred, root, attack.attack, "Image", atk_entr)

        except IndexError as e: 
            print("IndexError: Persistence Entropy values were all infinity")

        try: 
            org_entr = calc_entropy_model_CNN_stack(model, img, label, org_pred, root)
            add_persistence_entropy(model_name, ds_name, label, org_pred, root, "None", "CNN Output", org_entr)
                
        except IndexError as e:
            print("IndexError: Persistence Entropy values were all infinity")

        try: 
            atk_entr = calc_entropy_model_CNN_stack(model, atk_img, label, atk_pred, root, attack.attack)
            add_persistence_entropy(model_name, ds_name, label, atk_pred, root, attack.attack, "CNN Output", atk_entr)
                
        except IndexError as e:
            print("IndexError: Persistence Entropy values were all infinity")

        try: 
            img_wass_dist = calculate_wasserstein_distance(img.numpy(), atk_img.numpy(), label, org_pred, atk_pred, root, attack.attack)
            add_wasserstein_distance(model_name, ds_name, label, org_pred, atk_pred, root, attack.attack, "Image", img_wass_dist)

        except Exception as e: 
            print(f"Error calculting Wasserstein Distance: {e}")

        try: 
            cnn_wass_dist = calc_wass_dist_CNN_stack(model, img, atk_img, label, org_pred, atk_pred, root, attack.attack)
            add_wasserstein_distance(model_name, ds_name, label, org_pred, atk_pred, root, attack.attack, "CNN Output", cnn_wass_dist)

        except Exception as e: 
            print(f"Error calculting Wasserstein Distance: {e}")
        
def same_label_wasser_dist(model, img_map, model_name, ds_name, root):
    
    for label, imgs in img_map.items():

        for i in range(0, len(imgs) - 1, 2):
            img_1 = imgs[i]
            img_2 = imgs[i+1]

            try: 
                img_wasser_dist = calculate_wasserstein_distance(img_1.numpy(), img_2.numpy(), label, label, label, root, "None")
                add_wasserstein_distance(model_name, ds_name, label, label, label, root, "None", "Image", img_wasser_dist)
            except Exception as e: 
                print(f"Error calculating Wasserstein Distance: {e}")

            try: 
                cnn_wasser_dist = calc_wass_dist_CNN_stack(model, img_1, img_2, label, label, label, root, "None")
                add_wasserstein_distance(model_name, ds_name, label, label, label, root, "None", "CNN Output", cnn_wasser_dist)
            except Exception as e: 
                print(f"Error calculating Wasserstein Distance: {e}")

def sample_by_class(ds_test, correct_idxs, correct_predictions, incorrect_idxs, incorrect_predictions, force_even = False): 

    classes = np.unique(correct_predictions)

    inc_count_map = {num: 0 for num in classes}
    cor_img_map = {num: [] for num in classes}
    return_img_map = {num: [] for num in classes}

    for idx in incorrect_idxs: 
        inc_count_map[incorrect_predictions[idx]] += 1
    
    for idx in correct_idxs: 
        cor_img_map[correct_predictions[idx]].append(ds_test[idx][0])

    for num, count in inc_count_map.items():
        if force_even and count % 2 == 1: 
            count += 1
        return_img_map[num].extend(random.sample(cor_img_map[num], count))

    return return_img_map



def eval_model(model_save_path, model_name, dataset, root, num_classes, 
               make_tsne = True, make_persistence = True, make_avg_recons = False, atk_eps = 8/255, atk_eps_str = "8/255"):
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
    fgsm_attack = torchattacks.FGSM(lenet_model, eps = atk_eps)
    atk_model_output = query_model(lenet_model, dl_test, fgsm_attack, return_softmax = False)
    atk_predictions = outputs_to_predictions(torch.Tensor(atk_model_output))
    atk_correct_idxs, atk_incorrect_idxs = evaluate_dataset(model_name, ds_test.targets, atk_predictions, ds_name, root, True, f"{fgsm_attack.attack}({atk_eps_str})")

    if make_persistence: 
        # Two examples of persistent barcode with attacked misclassified point and its unattacked counterpart. 
        # One on the image and one on the output the CNN layers
        img, label = ds_test[atk_incorrect_idxs[0]]
        make_persistence_barcode(img.numpy(), label, root, False)
        barcode_model_CNN_Stack(lenet_model, img, label, root, False)
        atk_img = fgsm_attack(img.unsqueeze(0), torch.LongTensor([label])).squeeze(0).detach().cpu()
        make_persistence_barcode(atk_img.numpy(), label, root, True)
        barcode_model_CNN_Stack(lenet_model, atk_img, label, root, True)

        #Calculate Entropies
        make_persistence_metrics(lenet_model, ds_test, org_predictions, atk_predictions, model_name, ds_name, root, fgsm_attack)

        #Calculate Wasserstein distances of correct unattacked images
        img_map = sample_by_class(ds_test, org_correct_idxs, org_predictions, atk_incorrect_idxs, atk_predictions, force_even = True)
        same_label_wasser_dist(lenet_model, img_map, model_name, ds_name, root)


    if make_avg_recons and root == "data_original":
            img_map = sample_by_class(ds_test, org_correct_idxs, org_predictions, org_incorrect_idxs, org_predictions, False)

            file_name = f"hqa_0_10_recons_org_inc_no_atk.json"
            run_reconstruct_avg(lenet_model, file_name, ds_test, org_incorrect_idxs, 0, False, False)
            file_name = f"hqa_4_10_recons_org_inc_no_atk.json"
            run_reconstruct_avg(lenet_model, file_name, ds_test, org_incorrect_idxs, 4, False, False)
        
            file_name = f"hqa_0_10_recons_org_cor_no_atk.json"
            run_dict_reconstruct_avg(lenet_model, file_name, img_map, 0, False, False)
            file_name = f"hqa_4_10_recons_org_cor_no_atk.json"
            run_dict_reconstruct_avg(lenet_model, file_name, img_map, 4, False, False)

            file_name = f"hqa_0_10_recons_org_inc_atk_img.json"
            run_reconstruct_avg(lenet_model, file_name, ds_test, org_incorrect_idxs, 0, True, False)
            file_name = f"hqa_4_10_recons_org_inc_atk_img.json"
            run_reconstruct_avg(lenet_model, file_name, ds_test, org_incorrect_idxs, 4, True, False)
        
            file_name = f"hqa_0_10_recons_org_cor_atk_img.json"
            run_dict_reconstruct_avg(lenet_model, file_name, img_map, 0, True, False)
            file_name = f"hqa_4_10_recons_org_cor_atk_img.json"
            run_dict_reconstruct_avg(lenet_model, file_name, img_map, 4, True, False)

            file_name = f"hqa_0_10_recons_org_inc_atk_recons.json"
            run_reconstruct_avg(lenet_model, file_name, ds_test, org_incorrect_idxs, 0, False, True)
            file_name = f"hqa_4_10_recons_org_inc_atk_recons.json"
            run_reconstruct_avg(lenet_model, file_name, ds_test, org_incorrect_idxs, 4, False, True)
        
            file_name = f"hqa_0_10_recons_org_cor_atk_recons.json"
            run_dict_reconstruct_avg(lenet_model, file_name, img_map, 0, False, True)
            file_name = f"hqa_4_10_recons_org_cor_atk_recons.json"
            run_dict_reconstruct_avg(lenet_model, file_name, img_map, 4, False, True)

            file_name = f"hqa_0_10_recons_org_inc_atk_both.json"
            run_reconstruct_avg(lenet_model, file_name, ds_test, org_incorrect_idxs, 0, True, True)
            file_name = f"hqa_4_10_recons_org_inc_atk_both.json"
            run_reconstruct_avg(lenet_model, file_name, ds_test, org_incorrect_idxs, 4, True, True)
        
            file_name = f"hqa_0_10_recons_org_cor_atk_both.json"
            run_dict_reconstruct_avg(lenet_model, file_name, img_map, 0, True, True)
            file_name = f"hqa_4_10_recons_org_cor_atk_both.json"
            run_dict_reconstruct_avg(lenet_model, file_name, img_map, 4, True, True)
           
           

    # TSNE related
    if make_tsne and root in ["data_original", "data_recon_4"]:
        atk_mis_outputs = [atk_model_output[idx] for idx in atk_incorrect_idxs]
        atk_mis_true_labels = [ds_test.targets[idx] for idx in atk_incorrect_idxs]
        atk_mis_images = [ds_test[idx][0].squeeze(0).numpy().flatten() for idx in atk_incorrect_idxs]

        run_tsne(model_name, atk_mis_outputs, atk_mis_true_labels, ds_name, root, num_classes, "model output", fgsm_attack.attack, misclassified=True)
        run_tsne(model_name, atk_mis_images, atk_mis_true_labels, ds_name, root, num_classes, "input image", fgsm_attack.attack, misclassified=True)

        tsne_sample_idxs = org_correct_idxs

        if len(atk_incorrect_idxs) < len(org_correct_idxs):

            tsne_sample_idxs,_ = train_test_split(
                org_correct_idxs, 
                stratify = [ds_test[i][1] for i in org_correct_idxs],
                train_size = len(atk_incorrect_idxs),
                random_state = RANDOM_SEED
            )

        correct_outputs = [model_output[i] for i in tsne_sample_idxs]
        correct_images = [ds_test[i][0].squeeze(0).numpy().flatten() for i in tsne_sample_idxs]
        correct_targets = [ds_test[i][1] for i in tsne_sample_idxs]

        run_tsne(model_name, correct_outputs, correct_targets, ds_name, root, num_classes, "model output")
        run_tsne(model_name, correct_images, correct_targets, ds_name, root, num_classes, "input image")
    
    if make_persistence:
        pd.read_csv(PERS_ETP_OUTPUT_FILE, index_col=False).to_csv("persistence_entropies_copy.csv", index=False)
        pd.read_csv(WASS_DIST_OUTPUT_FILE, index_col=False).to_csv("wasserstein_distances_copy.csv", index=False)

def eval_tiled_model(model_save_path, model_name, dataset, root, num_classes, add_root = None, make_tsne = True, make_persistence = True, make_avg_recons = False):
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
    org_correct_idxs, _ = evaluate_dataset(model_name, ds_test._targets, org_predictions, ds_name, root_name, save_result=True)

    # make fgsm attack, query attacked model, evaluate the results
    fgsm_attack = torchattacks.FGSM(lenet_model)
    atk_model_output = query_model(lenet_model, dl_test, fgsm_attack, return_softmax = False)
    atk_predictions = outputs_to_predictions(torch.Tensor(atk_model_output))
    _, atk_incorrect_idxs = evaluate_dataset(model_name, ds_test._targets, atk_predictions, ds_name, root_name, True, fgsm_attack.attack)

    if make_persistence:
        #Two examples of persistent barcode with misclassified point and its attacked counterpart.
        # One on the image and one on the output the CNN layers
        img, label = ds_test[atk_incorrect_idxs[0]]
        make_persistence_barcode(img.numpy(), label, f"Tiled_{root_name}", False)
        barcode_model_CNN_Stack(lenet_model, img, label, f"Tiled_{root_name}", False)
        atk_img = fgsm_attack(img.unsqueeze(0), torch.LongTensor([label])).squeeze(0).detach().cpu()
        make_persistence_barcode(atk_img.numpy(), label, f"Tiled_{root_name}", True)
        barcode_model_CNN_Stack(lenet_model, atk_img, label, f"Tiled_{root_name}", True)

        #Calculate Entropies
        make_persistence_metrics(lenet_model, ds_test, org_predictions, atk_predictions, model_name, ds_name, root_name, fgsm_attack)

        img_map = sample_by_class(ds_test, org_correct_idxs, org_predictions, atk_incorrect_idxs, atk_predictions, force_even = True)
        
        same_label_wasser_dist(lenet_model, img_map, model_name, ds_name, root)

    if make_tsne and (add_root is not None or root in ["data_original", "data_recon_3"]):
        atk_mis_outputs = [atk_model_output[idx] for idx in atk_incorrect_idxs]
        atk_mis_images = [ds_test[idx][0].squeeze(0).numpy().flatten() for idx in atk_incorrect_idxs]
        mis_true_labels = [ds_test._targets[idx] for idx in atk_incorrect_idxs]

        run_tsne(model_name, atk_mis_outputs, mis_true_labels, ds_name, root_name, num_classes, "model output", fgsm_attack.attack, misclassified=True)
        run_tsne(model_name, atk_mis_images, mis_true_labels, ds_name, root_name, num_classes, "input image", fgsm_attack.attack, misclassified=True)

        tsne_sample_idxs = org_correct_idxs

        if len(atk_incorrect_idxs) < len(org_correct_idxs):

            tsne_sample_idxs, _ = train_test_split(
                org_correct_idxs, 
                stratify = [ds_test[i][1] for i in org_correct_idxs], 
                train_size = len(atk_incorrect_idxs),
                random_state = RANDOM_SEED
            )

        correct_outputs = [model_output[i] for i in tsne_sample_idxs]
        correct_images = [ds_test[i][0].squeeze(0).numpy().flatten() for i in tsne_sample_idxs]
        correct_targets = [ds_test[i][1] for i in tsne_sample_idxs]

        run_tsne(model_name, correct_outputs, correct_targets, ds_name, root_name, num_classes, "model output")
        run_tsne(model_name, correct_images, correct_targets, ds_name, root_name, num_classes, "input image")
    
    if make_persistence: 
        pd.read_csv(PERS_ETP_OUTPUT_FILE, index_col=False).to_csv("persistence_entropies_copy.csv", index=False)
        pd.read_csv(WASS_DIST_OUTPUT_FILE, index_col=False).to_csv("wasserstein_distances_copy.csv", index=False)
    
def main():

    for root in tqdm(RECON_ROOT_NAMES):

        eval_model(LENET_MNIST_PATH, "Lenet", IMG_MNIST_DIR_PATH, root, 10, make_tsne=False, make_persistence=False, make_avg_recons=False, atk_eps = 16/255, atk_eps_str = "16/255") # only model to calculate entropies\
        eval_model(LENET_ADV_MNIST_PATH, "Lenet (Adversarial)", IMG_MNIST_DIR_PATH, root, 10, make_tsne = True, make_persistence=False, atk_eps = 16/255, atk_eps_str = "16/255") # this experiment was only concerned about accuracy
        
        #eval_model(LENET_FASH_MNIST_PATH, "Lenet", IMG_FASH_MNIST_DIR_PATH, root, 10, make_tsne = True, make_persistence=False) 
        #eval_model(LENET_EMNIST_PATH, "Lenet", IMG_EMNIST_DIR_PATH, root, 47,  make_tsne = True, make_persistence=False) 
        #if root != "data_recon_4":
            #eval_model(LENET_MNIST_PATH, "Lenet", IMG_MNIST_FFT_DIR_PATH, root, 10)
            
    #        eval_tiled_model(LENET_MNIST_PATH, "Lenet", IMG_TILED_MNIST_DIR_PATH, root,  10,  make_tsne = True, make_persistence=False) 
    #        eval_tiled_model(LENET_FASH_MNIST_PATH, "Lenet", IMG_TILED_FASH_MNIST_DIR_PATH, root, 10,  make_tsne = True, make_persistence=False) 
    #        eval_tiled_model(LENET_EMNIST_PATH, "Lenet", IMG_TILED_EMNIST_DIR_PATH, root, 47,  make_tsne = True, make_persistence=False) 
        
    #eval_tiled_model(LENET_MNIST_PATH, "Lenet", IMG_TILED_MNIST_DIR_PATH, "data_recon_0", 10, "data_recon_3",  make_tsne = True, make_persistence=False)

if __name__ == "__main__": 
    main()