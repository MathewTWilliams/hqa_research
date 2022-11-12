# Author: Matt Williams
# Version: 10/29/2022


from LeNet import LeNet_5
from torchvision import transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader, random_split
from utils import device, MNIST_TRAIN_PATH, MNIST_TEST_PATH
import matplotlib.pyplot as plt
import torch
import numpy as np
from sklearn.metrics import classification_report

def run_lenet(early_stopping = False): 
    transform = transforms.Compose(
        [
            transforms.Resize(32), 
            transforms.CenterCrop(32), 
            transforms.ToTensor()
        ]
    )

    batch_size = 512

    ds_train = MNIST(MNIST_TRAIN_PATH, download=True, transform=transform)
    ds_train, ds_valid = random_split(ds_train, [50000,10000])

    dl_train = DataLoader(ds_train, batch_size=batch_size, shuffle = True, num_workers=4)
    dl_valid = DataLoader(ds_train, batch_size=batch_size, shuffle = True, num_workers=4)

    ds_test = MNIST(MNIST_TEST_PATH, download=True, train = False, transform=transform)
    dl_test = DataLoader(ds_test, batch_size=batch_size, shuffle = False, num_workers = 4)

    model = LeNet_5(dl_train, dl_valid, num_classes=10, early_stopping=early_stopping)
    model.to(device)

    train_losses, valid_losses = model.run_epochs(n_epochs=100, validate=True)

    
    plt.plot(range(1, len(train_losses) + 1), train_losses, label = "Training Loss")
    plt.plot(range(1, len(valid_losses) + 1), valid_losses, label = "Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Losses")
    plt.title("Training Loss values")
    plt.legend()
    plt.show()
    plt.clf()

    all_outputs = torch.Tensor().to(device)
    test_labels = []

    model.eval()
    with torch.no_grad(): 
        for data, labels in dl_test:
            cur_output = model(data.to(device))
            all_outputs = torch.cat((all_outputs, cur_output), 0)
            test_labels.extend(labels.tolist())

    softmax_probs = torch.exp(all_outputs).cpu()
    predictions = np.argmax(softmax_probs, axis = -1)

    class_report = classification_report(test_labels, predictions, output_dict = True)

    for key in class_report:
        print(key,":", class_report[key])

if __name__ == "__main__":
    run_lenet(True)
    run_lenet(False)
