#reference: https://github.com/JiahongChen/resnet-pytorch/blob/master/ResNet_FashionMNIST_Pytorch.ipynb

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import torchvision
import torchvision.datasets as datasets
import torchvision.models as models
from torchvision import transforms
import torch.optim as optim
import time
import tqdm as tqdm
from torch.autograd import Variable
from utils import FASH_MNIST_TRAIN_PATH, FASH_MNIST_TEST_PATH
from torchvision.models import ResNet18_Weights



class ResNetFeatureExtractor18(nn.Module):
    def __init__(self, pretrained = True):
        super(ResNetFeatureExtractor18, self).__init__()
        model_resnet18 = models.resnet18(weights = ResNet18_Weights.DEFAULT if pretrained else None)
        self.conv1 = model_resnet18.conv1
        self.bn1 = model_resnet18.bn1
        self.relu = model_resnet18.relu
        self.maxpool = model_resnet18.maxpool
        self.layer1 = model_resnet18.layer1
        self.layer2 = model_resnet18.layer2
        self.layer3 = model_resnet18.layer3
        self.layer4 = model_resnet18.layer4
        self.avgpool = model_resnet18.avgpool

        if not pretrained:
            weights_init(self.conv1)
            weights_init(self.bn1)
            weights_init(self.layer1)
            weights_init(self.layer2)
            weights_init(self.layer3)
            weights_init(self.layer4)

    def forward(self, x): 
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return x


class ResClassifier(nn.Module):
    def __init__(self, dropout_p = 0.5): #in_features = 512
        super(ResClassifier, self).__init__()
        self.fc = nn.Linear(512, 10)
    
    def forward(self, x): 
        out = self.fc(x)
        return out

def get_dataset():
    transform = transforms.Compose([transforms.ToTensor(),
                                    # expand channel from 1 to 3 to fit 
                                    # ResNet pretrained model
                                    transforms.Lambda(lambda x: x.repeat(3,1,1)),
                                    ])

    batch_size = 256
    #download dataset
    fash_mnist_train = datasets.FashionMNIST(root = FASH_MNIST_TRAIN_PATH, train=True, download=True, transform=transform)
    fash_mnist_test = datasets.FashionMNIST(root = FASH_MNIST_TEST_PATH, train=False, download=True, transform=transform)

    train_loader = torch.utils.data.DataLoader(fash_mnist_train, batch_size=batch_size, shuffle = True, num_workers = 0)
    test_loader = torch.utils.data.DataLoader(fash_mnist_test, batch_size=batch_size, shuffle = True, num_workers = 0)

    return train_loader, test_loader

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('conv') != -1:
        torch.nn.init.xavier_uniform_(m.weight)
    elif classname.find('Linear') != -1:
        torch.nn.init.xavier_uniform_(m.weight)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.01)


def test_accuracy(data_iter, netG, netF):
    """Evaluate testset accuracy of a model."""
    
    acc_sum, n = 0,0
    for (imgs, labels) in data_iter:
        # send data to the GPU if cuda is available
        if torch.cuda.is_available():
            imgs = imgs.cuda()
            labels = labels.cuda()
        netG.eval()
        netF.eval()
        with torch.no_grad():
            labels = labels.long()
            acc_sum += torch.sum((torch.argmax(netF(netG(imgs)), dim = 1) == labels)).float()
            n += labels.shape[0]
        return acc_sum.item()/n


def train(pretrained = True): 
    netG = ResNetFeatureExtractor18(pretrained=pretrained)

    netF = ResClassifier()

    if torch.cuda.is_available(): 
        netG = netG.cuda()
        netF = netF.cuda()

    opt_g = optim.SGD(netG.parameters(), lr = 0.01, weight_decay=0.0005)
    opt_f = optim.SGD(netF.parameters(), lr = 0.01, momentum = 0.9, weight_decay=0.00050)

    criterion = nn.CrossEntropyLoss()

    train_loader, test_loader = get_dataset()

    for epoch in range(10):
        n, start = 0, time.time()
        train_l_sum = torch.tensor([0.0], dtype = torch.float32)
        train_acc_sum = torch.tensor([0.0], dtype = torch.float32)

        for i, (imgs,labels) in tqdm.tqdm(enumerate(iter(train_loader))):
            netG.train()
            netF.train()
            imgs = Variable(imgs)
            labels = Variable(labels)
            #train on GPU if possible
            if torch.cuda.is_available(): 
                imgs = imgs.cuda()
                labels = labels.cuda()
                train_l_sum = train_l_sum.cuda()
                train_acc_sum = train_acc_sum.cuda()

            opt_g.zero_grad()
            opt_f.zero_grad()

            #extracted feature
            bottleneck = netG(imgs)

            #predicted labels
            label_hat = netF(bottleneck)

            #loss function
            loss = criterion(label_hat, labels)
            loss.backward()
            opt_g.step()
            opt_f.step()

            #calculate training error
            netG.eval()
            netF.eval()
            labels = labels.long()
            train_l_sum += loss.float()
            train_acc_sum += (torch.sum((torch.argmax(label_hat, dim = 1) == labels))).float()
            n += labels.shape[0]
        
        test_acc = test_accuracy(iter(test_loader), netG, netF)
        print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f, time %.1f sec'\
        % (epoch + 1, train_l_sum/n, train_acc_sum/n, test_acc, time.time() - start))


if __name__ == "__main__":
    #train(pretrained=True)
    train(pretrained=False)