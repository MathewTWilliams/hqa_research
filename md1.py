import os
import sys
import time
import argparse
from glob import glob
from pathlib import Path
import matplotlib
import matplotlib.image

import torch
import torch.nn as nn
import numpy as np
import random
from PIL import Image
import image_slicer
from torchvision import datasets, transforms
from datetime import datetime
from torchvision.datasets import MNIST
from torchvision.utils import make_grid
from torch.utils.data import Dataset, DataLoader


# EXERCISE IN TILING


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_data(train_data_dir, train_batch_size):
    
    train_transform = transforms.Compose([
        #transforms.Resize(256),
        #transforms.RandomResizedCrop(size=112),
        transforms.ToTensor(),
    ])

    train_dataset = datasets.ImageFolder(
        train_data_dir, 
        transform=train_transform
    )
    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    g = torch.Generator()
    g.manual_seed(0)
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset, 
        batch_size=train_batch_size, 
        shuffle=True,
        num_workers=8,
        drop_last=True,
        pin_memory=True,
        worker_init_fn=seed_worker,
        generator=g
    )
    return train_loader


def tile(filename, dir_in, dir_out, d):
    name, ext = os.path.splitext(filename)
    img = Image.open(os.path.join(dir_in, filename))
    w, h = img.size
    
    grid = product(range(0, h-h%d, d), range(0, w-w%d, d))
    for i, j in grid:
        box = (j, i, j+d, i+d)
        out = os.path.join(dir_out, f'{name}_{i}_{j}{ext}')
        img.crop(box).save(out)

def crop(path, im, label, height, width):
    imgwidth, imgheight = im.size 
    path_t = os.path.join(path, 'tiles')
    p = Path(path_t)
    p.mkdir(parents=True,exist_ok=True)

    k=0
    for i in range(0,imgheight,height):
        for j in range(0,imgwidth,width):
            box = (j, i, j+width, i+height)
            a = im.crop(box)
            a.save(os.path.join(path,"tiles",f"{label}IMG-{k}.png"))
            k +=1

transform = transforms.Compose(
    [
        transforms.Resize(32),
        transforms.CenterCrop(32),
        transforms.ToTensor(),
    ]
)
batch_size = 1
ds_train = MNIST('/tmp/mnist', download=True, transform=transform)
dl_train = DataLoader(ds_train, batch_size=batch_size, shuffle=True, num_workers=4)

ds_test = MNIST('/tmp/mnist_test_', download=True, train=False, transform=transform)
dl_test = DataLoader(ds_test, batch_size=batch_size, num_workers=4)
import ipdb; ipdb.set_trace()
test_x, test_label = next(iter(dl_test))
test_x = test_x.to(device)
test_label = test_label.cpu().numpy()[0]
test_x = test_x.squeeze().squeeze().cpu().numpy()
test_x = np.array(255*test_x, dtype = np.uint8)
path = os.path.join(os.getcwd(), 'data/', 'test', f'{test_label}')
p = Path(path)
p.mkdir(parents=True,exist_ok=True)
print(f"test_ image shape: {test_x.shape}")
#adjust for gray image here, either via gray map

matplotlib.image.imsave(p / f"img{test_label}orig.png", test_x) 
#im = Image.fromarray(test_x).convert("L")
#im.save(p / f"img{test_label}orig.png")
#testrecon = np.asarray(Image.open(p / f"img{test_label}orig.png").convert("L"))

im = Image.open(p / f"img{test_label}orig.png")#.convert("L")

imgwidth, imgheight = im.size

#tile one way

width=imgwidth//2
height=imgheight//2
crop(p, im, test_label, height, width)

#tike another way
tiles = image_slicer.slice(p / f"img{test_label}orig.png", 4)

new_tiles=[]
for tile in tiles:
	tile.image = Image.open(tile.filename)
	new_tiles.append(tile)
imrec = image_slicer.join(new_tiles)
imrec.save(p / f"img{test_label}tiled.png")
