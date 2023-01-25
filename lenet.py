# Author: Matt Williams
# Version 12/26/2022
import torch
from torch.nn import Linear, Conv2d, Sequential, Dropout, GELU
from torch.nn import Softmax, Flatten, MaxPool2d, ReLU, CrossEntropyLoss
from torch.optim import SGD
from pytorch_cnn_base import PyTorch_CNN_Base
import torch.nn.init as init

class Lenet_5(PyTorch_CNN_Base):
    """
    Classifier used for the various MNIST related datsets.
    """
    def __init__(self, train_loader, valid_loader, num_classes, save_path, stop_early = False):
        super(Lenet_5, self).__init__(train_loader,
                                    valid_loader,
                                    num_classes,
                                    save_path,
                                    stop_early)

    
    def _define_cnn_layers(self):
        conv_1 = Conv2d(in_channels=1, out_channels=6, kernel_size=5, padding = 2, stride=1)
        init.xavier_normal_(conv_1.weight)
        init.zeros_(conv_1.bias)

        conv_2 = Conv2d(in_channels=6, out_channels=16, kernel_size=5, padding = 0, stride=1)
        init.xavier_normal_(conv_2.weight)
        init.zeros_(conv_2.bias)

        conv_3 = Conv2d(in_channels=16, out_channels=120, kernel_size=5, padding = 0, stride=1)
        init.xavier_normal_(conv_3.weight)
        init.zeros_(conv_3.bias)

        cnn_layers = Sequential(
            conv_1, 
            #ReLU(inplace=True),
            GELU(), 
            MaxPool2d(kernel_size=2, stride = 2),
            conv_2, 
            #ReLU(inplace=True),
            GELU(),
            MaxPool2d(kernel_size=2, stride = 2),
            conv_3,
            #ReLU(inplace = True),
            GELU(),
            MaxPool2d(kernel_size=2, stride = 1)
        )

        return cnn_layers

    def _define_linear_layers(self):
        linear_1 = Linear(120, out_features=84)
        init.xavier_normal_(linear_1.weight)
        init.zeros_(linear_1.bias)

        linear_2 = Linear(84, out_features= self._num_classes)
        init.xavier_normal_(linear_2.weight)
        init.zeros_(linear_2.bias)

        linear_layers = Sequential(
            Flatten(),
            linear_1, 
            #ReLU(inplace=True),
            GELU(), 
            Dropout(),
            linear_2
            # not needed since Pytorch's implementation of CrossEntropyLoss
            # is LogSoftmax + Negative Log Likelihood loss which results 
            # in the same thing.
            #Softmax(dim = -1)
        )

        return linear_layers

    
    def _define_optimizer(self):
        return SGD(self.parameters(), lr = 0.01, momentum=0.9)

    def _define_loss_function(self):
        return CrossEntropyLoss()