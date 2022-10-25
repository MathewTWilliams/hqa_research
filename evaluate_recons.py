import pandas as pd
import numpy as np
from utils import MyDataset, PICKLED_RECON_PATH, LAYER_NAMES
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST

def main():
    columns = [layer_name for layer_name in LAYER_NAMES]
    columns.append("labels")
    recon_df = pd.DataFrame(data = pd.read_pickle(PICKLED_RECON_PATH), columns=columns)

    targets = recon_df["labels"]
    recon_df = recon_df.drop(columns="labels")
    #TODO: need to make dataset and loader for each layer
    recon_ds= MyDataset(data = recon_df, targets = targets)
    recon_dl = DataLoader(recon_ds)
    

    transform = transforms.Compose([
        transforms.Resize(32), 
        transforms.CenterCrop(32),
        transforms.ToTensor()
    ])

    test_ds = MNIST('/tmp/mnist_test_', download=True, train = False, transform=transform)
    test_dl = DataLoader(test_ds)
    

if __name__ == "__main__": 
    main()