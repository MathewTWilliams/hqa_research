import numpy as np
import datetime
import torch
import torch.nn.functional as F
from utils import get_bit_usage, device, LAYER_NAMES, LOG_DIR, MODELS_DIR
from r_adam import RAdam
from scheduler import FlatCA
import os
from hqa import HQA, VQCodebook
from torch.nn.utils import clip_grad_norm_


def decay_temp_linear(step, total_steps, temp_base, temp_min=0.001):
    factor = 1.0 - (step/total_steps)
    return temp_min + (temp_base - temp_min) * factor

def get_loss_hqa(img, model, epoch, step, commit_threshold=0.6, log=None):
    recon, orig, z_q, z_e, indices, KL, commit_loss = model(img)
    recon_loss = model.recon_loss(orig, recon)
    
    # calculate loss
    dims = np.prod(recon.shape[1:]) # orig_w * orig_h * num_channels
    loss = recon_loss/dims + 0.001*KL/dims + 0.001*(commit_loss)/dims
    
    # logging    
    if step % 20 == 0:
        nll = recon_loss
        elbo = -(nll + KL)  
        distortion_bpd = nll / dims / np.log(2)
        rate_bpd = KL / dims / np.log(2) # bits per dimension?
        
        bits, max_bits, highest_prob = get_bit_usage(indices)
        bit_usage_frac = bits / max_bits
        
        time = datetime.datetime.now().strftime("%d/%m/%Y %H:%M:%S")
        log_line = f"{time}, epoch={epoch}, step={step}, loss={loss:.5f}, distortion={distortion_bpd:.3f}, rate={rate_bpd:.3f}, -elbo={-elbo:.5f}, nll={nll:.5f}, KL={KL:.5f}, commit_loss={commit_loss:.5f}, bit_usage={bit_usage_frac:.5f}, highest_prob={highest_prob:.3f}, temp={model.codebook.temperature:.5f}"
        print(log_line)

        if log is not None:
            with open(log, "a") as logfile:
                logfile.write(log_line + "\n")
                
    return loss, indices


def train(dl_train, test_x, model, optimizer, scheduler, epochs, decay=True, log=None):
    step = 0
    model.train()
    temp_base = model.codebook.temperature
    code_count = torch.zeros(model.codebook.codebook_slots).to(device)
    total_steps = len(dl_train)*epochs
    
    for epoch in range(epochs):
        for x, _ in dl_train:
            x = x.to(device)
            
            # anneal temperature
            if decay is True:
                model.codebook.temperature = decay_temp_linear(step+1, total_steps, temp_base, temp_min=0.001) 
            
            loss, indices = get_loss_hqa(x, model, epoch, step, log=log)
                
            # take training step    
            model.zero_grad()
            loss.backward()
            clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            if scheduler is not None:
                scheduler.step()     
                
            # code reset every 20 steps
            indices_onehot = F.one_hot(indices, num_classes=model.codebook.codebook_slots).float()
            code_count = code_count + indices_onehot.sum(dim=(0, 1, 2))
            if step % 20 == 0:
                with torch.no_grad():
                    max_count, most_used_code = torch.max(code_count, dim=0)
                    frac_usage = code_count / max_count
                    z_q_most_used = model.codebook.lookup(most_used_code.view(1, 1, 1)).squeeze()

                    min_frac_usage, min_used_code = torch.min(frac_usage, dim=0)
                    if min_frac_usage < 0.03:
                        print(f'reset code {min_used_code}')
                        moved_code = z_q_most_used + torch.randn_like(z_q_most_used) / 100
                        model.codebook.codebook[min_used_code] = moved_code
                    code_count = torch.zeros_like(code_count)

            step += 1
        #if epoch % 5 == 0:
        #    for n in range(0, 5):
        #        show_recon(test_x[n, 0], model)
        #        plt.show();

def train_full_stack(dl_train, test_x, exp_name, epochs=5, lr=4e-4, layers = len(LAYER_NAMES)):
    
    enc_hidden_sizes = [16, 16, 32, 64, 128]
    dec_hidden_sizes = [16, 64, 256, 512, 1024]
    
    os.makedirs(LOG_DIR, exist_ok=True)
    
    layers = LAYER_NAMES[:min(layers, len(LAYER_NAMES))]

    for i,_ in enumerate(layers):
        print(f"training layer{i}")
        if i == 0:
            hqa = HQA.init_bottom(
                input_feat_dim=1,
                enc_hidden_dim=enc_hidden_sizes[i],
                dec_hidden_dim=dec_hidden_sizes[i],
            ).to(device)
        else:
            hqa = HQA.init_higher(
                hqa_prev,
                enc_hidden_dim=enc_hidden_sizes[i],
                dec_hidden_dim=dec_hidden_sizes[i],
            ).to(device)
        
        print(f"layer{i} param count {sum(x.numel() for x in hqa.parameters()):,}")
        
        log_file = os.path.join(LOG_DIR, f"{exp_name}_l{i}.log")
        opt = RAdam(hqa.parameters(), lr=lr)
        scheduler = FlatCA(opt, steps=epochs*len(dl_train), eta_min=lr/10)
        train(dl_train, test_x, hqa, opt, scheduler, epochs, log=log_file)
        hqa_prev = hqa
    
    torch.save(hqa, f"{MODELS_DIR}/{exp_name}.pt")
    return hqa