import torch
import random
import numpy as np
import os
import matplotlib
import matplotlib.pyplot as plt
import torch.nn.functional as F
import math
from PIL import Image
from pathlib import Path
from torchvision.utils import make_grid
from torch.utils.data import Dataset
import pandas as pd
from torchvision import transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LAYER_NAMES = ["Layer 0", "Layer 1", "Layer 2", "Layer 3", "Layer 4 Final"]

# File paths for downloading mnist related datasets
MNIST_TRAIN_PATH = '/tmp/mnist'
MNIST_TEST_PATH = '/tmp/mnist_test_'
FASH_MNIST_TRAIN_PATH = '/temp/fasion_mnist'
FASH_MNIST_TEST_PATH = '/temp/fasion_mnist_test_'
EMNIST_TRAIN_PATH = '/tmp/emnist'
EMNIST_TEST_PATH = '/tmp/emnist_test_'

HQA_MNIST_MODEL_NAME = "hqa_mnist_model"
HQA_FASH_MNIST_MODEL_NAME = "hqa_fash_mnist_model"
HQA_EMNIST_MODEL_NAME = "hqa_emnist_model"

CWD = os.path.abspath(os.getcwd())
PICKLED_DIR = os.path.join(CWD, "Pickle Files")

#Saved data file paths (original data and constructions)
IMG_DIR_PATH = os.path.join(CWD, "data")
IMG_MNIST_DIR_PATH = os.path.join(IMG_DIR_PATH, "MNIST")
IMG_FASH_MNIST_DIR_PATH = os.path.join(IMG_DIR_PATH, "Fashion_MNIST")
IMG_EMNIST_DIR_PATH = os.path.join(IMG_DIR_PATH, "EMNIST")
MNIST_PICKLED_RECON_PATH = os.path.join(PICKLED_DIR, "mnist_pickled_recons.pkl")
FASH_MNIST_PICKLED_RECON_PATH = os.path.join(PICKLED_DIR, "fash_mnist_pickled_recons.pkl")
EMNIST_PICKLED_RECON_PATH = os.path.join(PICKLED_DIR, "emnist_pickled_recons.pkl")

#Model relatd file paths
MODELS_DIR = os.path.join(CWD, "models")
HQA_MNIST_SAVE_PATH = os.path.join(MODELS_DIR, "hqa_mnist_model.pt")
HQA_FASH_MNIST_SAVE_PATH = os.path.join(MODELS_DIR, "hqa_fash_mnist_model.pt")
HQA_EMNIST_SAVE_PATH = os.path.join(MODELS_DIR, "hqa_emnist_model.pt")

TRAD_LENET_MNIST_PATH = os.path.join(MODELS_DIR, "trad_lenet_mnist.pt")
TRAD_LENET_FASH_MNIST_PATH = os.path.join(MODELS_DIR, "trad_lenet_fash_mnist.pt")
TRAD_LENET_EMNIST_PATH = os.path.join(MODELS_DIR, "trad_lenet_emnist.pt")
MOD_LENET_MNIST_PATH = os.path.join(MODELS_DIR, "mod_lenet_mnist.pt")
MOD_LENET_FASH_MNIST_PATH = os.path.join(MODELS_DIR, "mod_lenet_fash_mnist.pt")
MOD_LENET_EMNIST_PATH = os.path.join(MODELS_DIR, "mod_lenet_emnist.pt")

ACCURACY_OUTPUT_FILE = os.path.join(CWD, "classification_accuracies.csv")
ACCURACY_FILE_COLS = ["Model", "Dataset", "Attack", "Average"]
VISUAL_DIR = os.path.join(CWD, "Visuals")

MNIST_TRANSFORM = transforms.Compose([
    transforms.Resize(32),
    transforms.CenterCrop(32),
    transforms.ToTensor()
])

MNIST_BATCH_SIZE = 512

def access_results_df(filepath, columns): 
    results_df = pd.read_csv(filepath, index_col= False) \
                if os.path.exists(filepath) \
                else pd.DataFrame(columns=columns)
    return results_df


def combine_save_df(results_df, row_df, filepath):
    results_df = pd.concat([results_df, row_df], ignore_index=True)
    results_df.to_csv(filepath, index = False) 

def add_accuracy_results(model_name, dataset_name, attack_name, accuracies):
    accuracy_df = access_results_df(ACCURACY_OUTPUT_FILE, ACCURACY_FILE_COLS)

    row_dict = {"Model" : [model_name], 
                "Dataset" : [dataset_name], 
                "Attack" : [attack_name],
                "Average" : [np.sum(accuracies) / len(accuracies)]}

    row_df = pd.DataFrame(row_dict, columns=ACCURACY_FILE_COLS)
    combine_save_df(accuracy_df, row_df, ACCURACY_OUTPUT_FILE)

class MyDataset(Dataset):
    def __init__(self, data, targets, transform=None):
        self.data = data
        self.targets = torch.LongTensor(targets)
        self.transform = transform
        
    def __getitem__(self, index):
        x = self.data[index]
        y = self.targets[index]
        
        if self.transform:
            x = Image.fromarray(self.data[index] * 255, mode = "F")
            x = self.transform(x)

        return x, y
    
    def __len__(self):
        return len(self.data)

def set_seeds(seed=42, fully_deterministic=False):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    if fully_deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        

def show_image(im_data, scale=1):
    dpi = matplotlib.rcParams['figure.dpi']
    height, width = im_data.shape
    figsize = scale * width / float(dpi), scale * height / float(dpi)

    fig = plt.figure(figsize=figsize)
    ax = fig.add_axes([0, 0, 1, 1])

    # Hide spines, ticks, etc.
    ax.axis('off')
    ax.imshow(im_data, vmin=0, vmax=1, cmap='gray')
    plt.show()
    ax.set(xlim=[0, width], ylim=[height, 0], aspect=1)


# TRAINING
def show_recon(img, *models):
    fig, axes = plt.subplots(nrows=1, ncols=len(models), figsize=(10 * len(models), 5))

    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])

    for i, model in enumerate(models):
        model.eval()
        img_ = img.unsqueeze(0).unsqueeze(0)
        recon = model.reconstruct(img_).squeeze()
        output = np.hstack([img.cpu(), np.ones([img.shape[0], 1]), recon.cpu(), np.ones([img.shape[0], 1]), np.abs((img-recon).cpu())])
        axes[i].imshow(output, vmin=0, vmax=1, cmap='gray')
        model.train()

#TRAINING
def get_bit_usage(indices):
    """ Calculate bits used by latent space vs max possible """
    num_latents = indices.shape[0] * indices.shape[1] * indices.shape[2]
    avg_probs = F.one_hot(indices).float().mean(dim=(0, 1, 2))
    highest_prob = torch.max(avg_probs)
    bits = (-(avg_probs * torch.log2(avg_probs + 1e-10)).sum()) * num_latents
    max_bits = math.log2(256) * num_latents
    return bits, max_bits, highest_prob


# TRAINING
def save_img(recon, label, path, idx):
	p = Path(path)
	p.mkdir(parents=True,exist_ok=True)
	print(f"recon image shape: {recon.shape}")
	matplotlib.image.imsave(p / f"img{label}_{idx}.png", recon.cpu().numpy(), cmap = "gray")	
	checkrecon = np.asarray(Image.open(p / f"img{label}_{idx}.png").convert("L"))
	print(f"loaded image shape: {checkrecon.shape}")


# LAYERS RECONSTRUCTION
def recon_comparison(model, ds_test, names, descriptions, img_save_dir):
    images = []
    targets = []
    for idx in range(len(ds_test)):
        (image, label) = ds_test[idx]    
        img = image.to(device).squeeze()
        images.append(img.cpu().numpy())
        targets.append(label)
    #import ipdb; ipdb.set_trace()
    print("Original images to be reconstructed")
    output = np.hstack(images)
    #show_image(output)
    
    output_dict = {name:[] for name in names}

    for layer, name, description in zip(model, names, descriptions):
        images = []
        
        for idx in range(len(ds_test)):
            (image, label) = ds_test[idx]    
            img = image.to(device).squeeze()
            
            for_recon = img.unsqueeze(0).unsqueeze(0)
            layer.eval()
            recon = layer.reconstruct(for_recon).squeeze()
            output_dict[name].append(recon.cpu().numpy())
            images.append(recon.cpu().numpy()) 

            recon_path = os.path.join(img_save_dir, f"data_recon_{names.index(name)}", f'{label}')
            orig_path = os.path.join(img_save_dir, 'data_original', f'{label}')
            jpg_path = os.path.join(img_save_dir, "data_jpg", f"{label}")
            save_img(recon, label, recon_path, idx)
            save_img(img, label, orig_path, idx)

            im = Image.open(os.path.join(orig_path, f"img{label}_{idx}.png"))
            #print("The size of the image before conversion :", end = "")
            #print(os.path.getsize(os.path.join(orig_path, f"img{label}_{idx}.png")))

            #converting to jpg
            gr_im = im.convert("L")

            #exporting the image
            p = Path(jpg_path)
            p.mkdir(parents = True, exist_ok=True)

            gr_im.save(os.path.join(jpg_path, f"img{label}_{idx}.jpeg"))
            #print("The size of the image after conversion : ", end = "")
            #print(os.path.getsize(os.path.join(jpg_path, f"img{label}_{idx}.jpeg")))
    
        print(f"{name}: {description}")
        output = np.hstack(images)
        #show_image(output)

    return output_dict, targets

# HQA distortions in Fig 3

def get_rate_upper_bound(model, example_input):
    assert len(example_input.shape) == 4, "Expected (1, num_channels, x_h, x_w)"
    assert example_input.shape[0] == 1, "Please provide example with batch_size=1"
    
    z_e = model.encode(example_input)
    _, top_indices, _, _ = model.codebook(z_e)
        
    # assume worst case scenario: we have a uniform usage of all our codes
    rate_bound = top_indices[0].numel() * np.log2(model.codebook.codebook_slots)

    return rate_bound

def test(model, dl_test):
    model.eval()
    total_nll = []
    total_kl = []
    
    for x, _ in dl_test:
        img = x.to(device)       
        recon, orig, z_q, z_e, indices, kl, _ = model.forward_full_stack(img)       
        recon_loss = model[0].recon_loss(img, recon)        
        total_nll.append(recon_loss.item())
        if kl != 0:
            total_kl.append(kl.item())
        else:
            total_kl.append(kl)
    
    dims = np.prod(x.shape[1:])
    kl_mean = np.mean(total_kl)
    nll_mean = np.mean(total_nll)
    distortion_bpd = nll_mean / dims / np.log(2)
    rate_bpd = kl_mean / dims / np.log(2)
    elbo = -(nll_mean + kl_mean)
    
    rate_bound = get_rate_upper_bound(model, img[0].unsqueeze(0))
    
    return distortion_bpd, rate_bound


def get_rd_data(model, dl_test):
    dist = []
    rates = []
    
    for i, _ in enumerate(model):
        d, r = test(model[i], dl_test)
        dist.append(float(d))
        rates.append(float(r))
    
    return dist, rates

# Layer-wise interpolations

def interpolate(a, b, ds_test, vqvae, grid_x=16):
    images = []
    
    x_a,_ = ds_test[a]
    x_b,_ = ds_test[b]
    point_1 = vqvae.encode(x_a.unsqueeze(0).to(device))
    point_2 = vqvae.encode(x_b.unsqueeze(0).to(device))

    interpolate_x = np.linspace(point_1[0].cpu().numpy(), point_2[0].cpu().numpy(), grid_x)
    
    results = torch.Tensor(len(interpolate_x), 1, 32, 32)
    for i, z_e_interpolated in enumerate(interpolate_x):       
        z_e = torch.Tensor(z_e_interpolated).unsqueeze(0).to(device)
        z_q = vqvae.quantize(z_e)
        recon = vqvae.decode(z_q).squeeze() 
        results[i] = recon

    grid_img = make_grid(results.cpu(), nrow=grid_x)
    show_image(grid_img[0,:,:])

def show_original(idx, ds_test):
    x, _ = ds_test[idx]
    image = x.squeeze()
    show_image(image)
    

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
