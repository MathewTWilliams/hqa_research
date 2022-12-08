# Author: Matt Williams
# Version: 12/6/2022

import torch
from torch.nn import Linear, Conv2d, Sequential, Dropout
from torch.nn import Module, Softmax, CrossEntropyLoss, Flatten
from utils import device, MODELS_DIR
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F
import os

class LeNet_5(Module):
   
    def __init__(self, train_loader, valid_loader, num_classes, loss_function, Pooling_Type,
                 Activ_Func_Type, save_path, add_normalization = False, pad_first_conv = True, 
                 early_stopping = False):
        '''
        Args:
        - train_loader: instance of DataLoader containing our training data.
        - valid_loader: instance of DataLoader containing out validation data.
        - num_classes: how many classes are contained in the data.
        - loss_function: an instance of the loss function to use.
        - pooling_type: a reference (not instance) to the type of pooling layer to use (e.g. MaxPool2d, AvgPool2d).
        - activ_func: a reference (not instance) to the type of activation function to use.
        - save_path: file path to save this model after an epoch.
        - pad_first_conv: boolean to determine if the first convolution layer should be padded or not. If
            the first convolution layer is padded, then an additional pooling layer is added.
        - early_stopping: should the model stop training early if validation loss increases.
        ''' 
        super(LeNet_5, self).__init__()
        self.__train_loader = train_loader
        self.__valid_loader = valid_loader
        self.__num_classes = num_classes
        self.__early_stopping = early_stopping
        self.__loss_function = loss_function.to(device)
        self.__save_path = save_path

        if early_stopping: 
            new_file_name = "early_" + save_path.split("\\")[-1]
            self.__save_path = os.path.join(MODELS_DIR, new_file_name)

        self.__init_model(Pooling_Type, Activ_Func_Type, pad_first_conv)

    def __init_model(self, Pooling_Type, Activ_Fun, pad_first_conv): 
        
        first_padding = 2 if pad_first_conv else 0

        conv_1 = Conv2d(in_channels=1, out_channels=6, kernel_size=5, padding = first_padding, stride = 1)
        torch.nn.init.xavier_uniform_(conv_1.weight)
        torch.nn.init.zeros_(conv_1.bias)

        conv_2 = Conv2d(in_channels=6, out_channels=16, kernel_size = 5, padding = 0, stride = 1)
        torch.nn.init.xavier_uniform_(conv_2.weight)
        torch.nn.init.zeros_(conv_2.bias)

        conv_3 = Conv2d(in_channels=16, out_channels=120, kernel_size = 5, padding = 0, stride = 1)
        torch.nn.init.xavier_uniform_(conv_3.weight)
        torch.nn.init.zeros_(conv_3.bias)

        self.__cnn_layers = Sequential(
            conv_1,
            Activ_Fun(),
            Pooling_Type(kernel_size=2, stride = 2),
            conv_2,
            Activ_Fun(),
            Pooling_Type(kernel_size = 2, stride = 2),
            conv_3, 
            Activ_Fun(),
        )

        if pad_first_conv:
            self.__cnn_layers.append(Pooling_Type(kernel_size = 2, stride = 1))

        linear_1 = Linear(120, out_features=84)
        torch.nn.init.xavier_uniform_(linear_1.weight)
        torch.nn.init.zeros_(linear_1.bias)

        linear_2 = Linear(84, out_features=self.__num_classes)
        torch.nn.init.xavier_uniform_(linear_2.weight)
        torch.nn.init.zeros_(linear_2.bias)

        self.__linear_layers = Sequential(
            Flatten(),
            linear_1,
            Dropout(),
            Activ_Fun(),
            linear_2, 
        )

        if not isinstance(self.__loss_function, CrossEntropyLoss):
            self.__linear_layers.append(Softmax(dim = -1))


    def set_optimizer(self, optimizer): 
        self.__optimizer = optimizer

    def forward(self, x):
        x = self.__cnn_layers(x)
        x = self.__linear_layers(x)
        return x

    def run_epochs(self, n_epochs, validate = True):
        train_losses = []
        valid_losses = []
        min_valid_loss = np.inf

        for _ in tqdm(range(n_epochs)):
            train_loss = self.__train()
            train_losses.append(train_loss)

            if validate:
                valid_loss = self.__validate()
                valid_losses.append(valid_loss)

                if self.__early_stopping and valid_loss > min_valid_loss: 
                    break
                
                min_valid_loss = valid_loss

            torch.save(self, self.__save_path)

        return train_losses, valid_losses

    def __train(self):
        training_loss = 0
        self.train(True)
        for data, labels in self.__train_loader:
            self.train()
            data = data.to(device)
            if not isinstance(self.__loss_function, CrossEntropyLoss):
                labels = F.one_hot(labels).float()
            labels = labels.to(device)

            self.__optimizer.zero_grad()
            output = self(data)
            loss = self.__loss_function(output, labels)

            loss.backward()
            self.__optimizer.step()
            training_loss += loss.item()
        
        return training_loss / len(self.__train_loader)

    def __validate(self):
        valid_loss = 0
        self.eval()

        for data, labels in self.__valid_loader:
            data = data.to(device)
            labels = labels.to(device)

            with torch.no_grad():
                output = self(data)
                loss = self.__loss_function(output, labels)
                valid_loss += loss.item()

        return valid_loss / len(self.__valid_loader)

