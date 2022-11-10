# Author: Matt Williams
# Version: 10/28/2022

import torch
from torch.nn import Linear, Conv2d, AvgPool2d, Sequential
from torch.nn import Module, Tanh, Softmax, CrossEntropyLoss
from torch.optim import SGD
from utils import device
import numpy as np
from utils import LENET_SAVE_PATH
from tqdm import tqdm

class LeNet_5(Module): 
    def __init__(self, train_loader, valid_loader, num_classes): 
        super(LeNet_5, self).__init__()
        self.__train_loader = train_loader
        self.__valid_loader = valid_loader
        self.__num_classes = num_classes
        self.__init_model()
        

    def __init_model(self): 
        
        conv_1 = Conv2d(in_channels=1, out_channels=6, kernel_size=5, padding = 2, stride = 1)
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
            Tanh(), 
            AvgPool2d(kernel_size=2, stride = 2),
            conv_2,
            Tanh(),
            AvgPool2d(kernel_size = 2, stride = 2),
            conv_3, 
            Tanh(),
            AvgPool2d(kernel_size=2, stride = 1)
        )

        linear_1 = Linear(120, out_features=84)
        torch.nn.init.xavier_uniform_(linear_1.weight)
        torch.nn.init.zeros_(linear_1.bias)

        linear_2 = Linear(84, out_features=self.__num_classes)
        torch.nn.init.xavier_uniform_(linear_2.weight)
        torch.nn.init.zeros_(linear_2.bias)

        self.__linear_layers = Sequential(
            linear_1,
            Tanh(),
            linear_2 
            #Softmax(dim=-1)
            # No Softmax layer at the end
            # Py-Torch's implementation of Cross Entropy loss uses
            # LogSoftmax with negative log likelihood loss to acheive the
            # same results
        )

        self.__optimizer = SGD(self.parameters(), lr = 0.01, momentum = 0.9)

        self.__loss_function = CrossEntropyLoss().to(device)

    def forward(self, x):
        x = self.__cnn_layers(x)
        x = x.view(x.size(0), -1)
        x = self.__linear_layers(x)
        return x

    def run_epochs(self, n_epochs, validate = True, early_stopping = False):
        train_losses = []
        valid_losses = []
        min_valid_loss = np.inf

        for _ in tqdm(range(n_epochs)):
            train_loss = self.__train()
            train_losses.append(train_loss)

            if validate:
                valid_loss = self.__validate()
                valid_losses.append(valid_loss)

                if early_stopping and valid_loss > min_valid_loss: 
                    break
                
                min_valid_loss = valid_loss
                torch.save(self, LENET_SAVE_PATH)

        return train_losses, valid_losses

    def __train(self):
        training_loss = 0
        self.train(True)
        for data, labels in self.__train_loader:
            data = data.to(device)
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

