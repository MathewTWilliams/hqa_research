from load_datasets import load_mnist
import matplotlib.pyplot as plt
from slice_torch_dataset import CombinedDataSet
from numeric_image_folder import NumericImageFolder
from utils import *
from evaluate_recons import img_folder_transform
from torch.utils.data import DataLoader
if __name__ == "__main__":

    ds_test = NumericImageFolder(os.path.join(IMG_TILED_MNIST_DIR_PATH, "data_recon_0"), transform = img_folder_transform)
    ds_test = CombinedDataSet(ds_test, num_tiles=2, tile_split="v")
   
    plt.imshow(ds_test[0][0].squeeze(0), cmap="gray")
    plt.show()
    plt.clf()

    
    
