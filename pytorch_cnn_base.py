# Author: Matt Williams
# Version: 1/10/2022


import torch
from torch.nn import Module
import numpy as np
from tqdm import tqdm
from utils import device, MODELS_DIR
import torch.nn.functional as F
import os
import matplotlib.pyplot as plt
from persistent_homology import make_persistence_barcode, make_vectorized_persistence
from contextlib import nullcontext


class PyTorch_CNN_Base(Module):
    """Base class for all Py_Torch models"""
    def __init__(self, train_loader, valid_loader, num_classes, save_path, stop_early = False):
        """
        Arguments: 
        - train_loader: DataLoader for training set. 
        - valid_loader: DataLoader for validation set. Can be None.
        - num_classes: Number of classes in the dataset.
        - save_path: File path at which to save the model after training.
        - stop_early: should training stop early if the validation loss increases?   
        """

        super(PyTorch_CNN_Base, self).__init__()
        self._train_loader = train_loader
        self._valid_loader = valid_loader
        self._num_classes = num_classes
        self._save_path = save_path
        self._stop_early = stop_early

        self._cnn_layers = self._define_cnn_layers()
        self._linear_layers = self._define_linear_layers()
        self._optimizer = self._define_optimizer()
        self._loss_function = self._define_loss_function()
        self._loss_function = self._loss_function.to(device)

        if stop_early:
            new_file_name = "early_" + save_path.split("\\")[-1]
            self._save_path = os.path.join(MODELS_DIR, new_file_name)

        self = self.to(device)
        
    def _define_cnn_layers(self):
        '''Needs to be defned by the super class'''
        raise NotImplementedError()

    def _define_linear_layers(self):
        '''Needs to be defned by the super class'''
        raise NotImplementedError()

    def _define_optimizer(self):
        '''Needs to be defned by the super class'''
        raise NotImplementedError()

    def _define_loss_function(self):
        '''Needs to be defned by the super class'''
        raise NotImplementedError()

    def forward(self, x):
        x = self._cnn_layers(x)
        x = self._linear_layers(x)
        return x

    def run_epochs(self, n_epochs, validate = True, attack = None):
        train_losses = []
        valid_losses = []
        min_valid_loss = np.inf

        for _ in (tqdm(range(n_epochs))):
            train_loss = self._train(attack)
            train_losses.append(train_loss)

            if validate and self._valid_loader is not None:
                valid_loss = self._validate(attack)
                valid_losses.append(valid_loss)

                if self._stop_early and valid_loss > min_valid_loss:
                    break

                min_valid_loss = valid_loss
            torch.save(self, self._save_path)
        
        return train_losses, valid_losses

    def _train(self, attack):
        training_loss = 0
        self.train()

        for data, labels in self._train_loader:
            data = data.to(device)

            if attack: 
                adv_data = attack(data, labels)
                data = torch.cat((data.to(device), adv_data), axis = 0)
                labels = torch.cat([labels, labels], axis = 0 )

            labels = F.one_hot(labels, num_classes = self._num_classes).float()
            labels = labels.to(device)

            self._optimizer.zero_grad()
            output = self(data)
            loss = self._loss_function(output, labels)
            loss.backward()
            self._optimizer.step()
            training_loss += loss.item()

        return training_loss / len(self._train_loader)

    def _validate(self, attack):
        valid_loss = 0
        self.eval()

        for data, labels in self._valid_loader:
            data = data.to(device)

            if attack: 
                adv_data = attack(data, labels)
                data = data.to(device)
                data = torch.cat((data, adv_data), axis = 0)
                labels = torch.cat([labels, labels], axis = 0)

            labels = F.one_hot(labels, num_classes = self._num_classes).float()
            labels = labels.to(device)

            context_manager = torch.no_grad()
            if attack: 
                context_manager = nullcontext()

            with context_manager:
                output = self(data)
                loss = self._loss_function(output, labels)
                valid_loss += loss.item()

        return valid_loss / len(self._valid_loader)


def outputs_to_predictions(model_output):
    softmax_probs = torch.exp(model_output).numpy()
    return np.argmax(softmax_probs, axis = -1)

def query_model(model, dl_test, attack = None, return_softmax = True):
    outputs = torch.Tensor().to(device)
    
    for data, labels in dl_test:
        if attack is not None: 
            data = attack(data, labels)
        cur_output = model(data.to(device))
        outputs = torch.cat((outputs, cur_output), axis = 0)
    
    outputs = outputs.detach().cpu()
    if return_softmax:
        outputs = outputs_to_predictions(outputs)

    else: 
        outputs = outputs.numpy()

    return outputs
