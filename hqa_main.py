import torch
import os
from torchvision import transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from training import train_full_stack
from utils import *
from torchvision.utils import make_grid
import numpy as np
from hqa import *
import pandas as pd


def main():
    
    print(f"CUDA={torch.cuda.is_available()}", os.environ.get("CUDA_VISIBLE_DEVICES"))
    #z = np.random.rand(5, 5)
    #plt.imshow(z)
    
    #MNIST DATASETS
    
    transform = transforms.Compose(
        [
            transforms.Resize(32),
            transforms.CenterCrop(32),
            transforms.ToTensor(),
        ]
    )
    batch_size = 512
    ds_train = MNIST(MNIST_TRAIN_PATH, download=True, transform=transform)
    dl_train = DataLoader(ds_train, batch_size=batch_size, shuffle=True, num_workers=4)
    
    
    ds_test = MNIST(MNIST_TEST_PATH, download=True, train=False, transform=transform)
    dl_test = DataLoader(ds_test, batch_size=batch_size, num_workers=4)
    test_x, _ = next(iter(dl_test))
    test_x = test_x.to(device)
    
    # TRAIN HQA STACK
    
    # Train a HQA stack
    os.makedirs(MODELS_DIR, exist_ok=True)
    
    
    if not os.path.isfile(HQA_SAVE_PATH):
        hqa_model = train_full_stack(dl_train, test_x, MODELS_DIR, HQA_MODEL_NAME, epochs=5)
    else:
        hqa_model = torch.load(HQA_SAVE_PATH)
    
    hqa_model.eval()
        
    layer_descriptions = [
        "downsample 2 in each dimension, latent space size of 16x16",
        "downsample 4 in each dimension, latent space size of 8x8",
        "downsample 8 in each dimension, latent space size of 4x4",
        "downsample 16 in each dimension, latent space size of 2x2",
        "downsample 32 in each dimension, latent space size of 1x1",
    ]

    # Show reconstruction comparison over each layer in HQA
    output_dict, targets = recon_comparison(hqa_model, ds_test, LAYER_NAMES, layer_descriptions)
    
    
    # Layer distortions
    distortions, rates = get_rd_data(hqa_model,dl_test)
    print("Name \t\t Distortion \t Rate")
    for dist, rate, name in zip(distortions, rates, LAYER_NAMES):
        print(f"{name} \t {dist:.4f} \t {int(rate)}")
    
    # Free samples
    num_codes = hqa_model.codebook.codebook_slots
    results = torch.Tensor(num_codes, 1, 32, 32).to(device)
    count=0
    for i in range(num_codes):
        codes = torch.LongTensor([i]).unsqueeze(0).unsqueeze(0).to(device)
        results[count] = hqa_model.reconstruct_from_codes(codes)
        count += 1
            
    grid_img = make_grid(results.cpu(), nrow=16)
    #show_image(grid_img[0,:,:])
    
    # Final layer interpolations
    
    grid_x = grid_y = 16
    results = torch.Tensor(grid_x * grid_y, 1, 32, 32)
    i = 0
    
    for j in range(grid_y):
        x_a,_ = ds_test[j]
        x_b,_ = ds_test[j+grid_y]
        point_1 = hqa_model.encode(x_a.unsqueeze(0).to(device)).cpu()
        point_2 = hqa_model.encode(x_b.unsqueeze(0).to(device)).cpu()
        interpolate_x = np.linspace(point_1[0], point_2[0], grid_x)
    
        for z_e_interpolated in interpolate_x:
            z_e_i = torch.Tensor(z_e_interpolated).unsqueeze(0).to(device)
            z_q = hqa_model.quantize(z_e_i)
            results[i] = hqa_model.decode(z_q).squeeze()
            i += 1
                
    grid_img = make_grid(results.cpu(), nrow=grid_x)
    #       show_image(grid_img[0,:,:])
    
    #Stochastic Reconstructions
    
    # Show held-out reconstructions: [ORIG, 14xSAMPLE, AVERAGED_10_SAMPLES]
    grid_x = grid_y = 16
    results = torch.Tensor(grid_x * grid_y, 1, 32, 32)
    
    result_idx = 0
    for test_idx in range(grid_y):
        x_a,_ = ds_test[test_idx]
        img = x_a.squeeze().to(device)
        img_ = img.unsqueeze(0).unsqueeze(0)
        num_examples = 5
        
        # ORIG
        results[result_idx] = img
        result_idx += 1
        
        # 14 RANDOM STOCHASTIC DECODES
        for _ in range(grid_x -2):
            results[result_idx] = hqa_model.reconstruct(img_).squeeze()
            result_idx += 1
        
        # AVERAGED SAMPLES
        results[result_idx] = hqa_model.reconstruct_average(img_, num_samples=14).squeeze()
        result_idx += 1
    
    grid_img = make_grid(results.cpu(), nrow=grid_x)
    show_image(grid_img[0,:,:])
    
    # Layer-wise interpolations
    print("Originals")
    show_original(1, ds_test)
    show_original(9, ds_test)
    for layer, name, description in zip(hqa_model, LAYER_NAMES, layer_descriptions):
        print(f"{name} : {description}")
        interpolate(1, 9, ds_test, layer, grid_x=10)
        
    #     TEST DATASET CREATION AND PICKLING
    # Let's create 10 RGB images of size 128x128 and 10 labels {0, 1}
    # data = list(np.random.randint(0, 255, size=(10, 3, 32, 32)))
    # targets = list(np.random.randint(2, size=(10)))
    
    # transform = transforms.Compose([transforms.Resize(64), transforms.ToTensor()])
    # dataset = MyDataset(data, targets, transform=transform)
    # im_test = dataset[5]
    # dataloader = DataLoader(dataset, batch_size=5)

    recon_data = [value for _ , value in output_dict.items()]
    columns = [name for name in LAYER_NAMES]
    columns.append("labels")
    recon_df = pd.DataFrame(data = list(zip(*recon_data, targets)), columns=columns)
    recon_df.to_pickle(PICKLED_RECON_PATH)


    

if __name__ == "__main__":
    set_seeds()
    main()
