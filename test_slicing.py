from load_datasets import load_mnist
import matplotlib.pyplot as plt
from slice_torch_dataset import CombinedDataSet



if __name__ == "__main__":
    _, _, dl_test = load_mnist(return_tiled=True, num_tiles=4)

    img_1 = dl_test.dataset[0][0].squeeze(0)
    img_2 = dl_test.dataset[1][0].squeeze(0)
    img_3 = dl_test.dataset[2][0].squeeze(0)
    img_4 = dl_test.dataset[3][0].squeeze(0)
    
    plt.imshow(img_1, cmap = "gray")
    plt.show()
    plt.clf()

    plt.imshow(img_2, cmap = "gray")
    plt.show()
    plt.clf()

    plt.imshow(img_3, cmap = "gray")
    plt.show()
    plt.clf()

    plt.imshow(img_4, cmap = "gray")
    plt.show()
    plt.clf()

    combined_ds = CombinedDataSet(dl_test.dataset, num_tiles=4)

    combined_img = combined_ds[0][0].squeeze(0)
    plt.imshow(combined_img, cmap="gray")
    plt.show()
    plt.clf()