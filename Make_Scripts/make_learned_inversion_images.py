import numpy as np
import torch
import matplotlib.pyplot as plt

import os

import sys
sys.path.append('./')
from networks import *

from cil.utilities.display import show2D


if __name__ == "__main__":
    save_folder = "Learned_Inversion_Results/"
    try:
        os.mkdir(save_folder)
    except:
        pass
    dev = torch.device('cuda:0')
    net_brain = LearndInversion().to(dev)
    net_brain.load_state_dict(torch.load('State_Dictionaries/brain_learned_inversion'))
    net_brain.eval()

    net_chest = LearndInversion().to(dev)
    net_chest.load_state_dict(torch.load('State_Dictionaries/chest_learned_inversion'))
    net_chest.eval()

    net_both = LearndInversion().to(dev)
    net_both.load_state_dict(torch.load('State_Dictionaries/both_learned_inversion'))
    net_both.eval()

    try:
        os.mkdir(save_folder + "chest/")
    except:
        pass
    y_true = np.load("Datasets/chest_testing.npy").astype(np.float32)
    z = np.load("Datasets/chest_testing_recon.npy").astype(np.float32)

    d = np.load("Datasets/chest_testing_data_limited.npy").astype(np.float32)

    x = np.load("Datasets/chest_testing_recon_limited.npy").astype(np.float32)

    X = torch.from_numpy(d).to(dev)
    X = torch.unsqueeze(X,1)

    with torch.no_grad():
        Y_chest = net_chest(X)
        y_chest = np.squeeze(Y_chest.cpu().detach().numpy())
        Y_brain = net_brain(X)
        y_brain = np.squeeze(Y_brain.cpu().detach().numpy())
        Y_both = net_both(X)
        y_both = np.squeeze(Y_both.cpu().detach().numpy())
    rmse_full = np.mean((z- y_true)**2,axis = (1,2))**(1/2)
    rmse_limited = np.mean((x- y_true)**2,axis = (1,2))**(1/2)
    rmse_chest = np.mean((y_chest - y_true)**2, axis = (1,2))**(1/2)
    rmse_brain = np.mean((y_brain - y_true)**2, axis = (1,2))**(1/2)
    rmse_both = np.mean((y_both - y_true)**2, axis = (1,2))**(1/2)

    rmses = [rmse_full, rmse_limited, rmse_chest, rmse_brain, rmse_both]
    fig,ax = plt.subplots(1,1)
    ax.violinplot(rmses,
                  showmeans=False,
                  showmedians=True)
    
    ax.set_xticks([y + 1 for y in range(len(rmses))],
                  labels=['Full', 'Limited', 'Chest', 'Brain', 'Both'])
    plt.savefig(save_folder + "chest/hist.png")
    plt.close()

    for j in range(x.shape[0]):
        show2D([y_true[j,:,:],  z[j,:,:], x[j,:,:], y_chest[j,:,:], y_brain[j,:,:], y_both[j,:,:]], title = ["True", "Full Recon", "Limited Recon", "Chest Learned Inversion", "Brain Learned Inversion", "Both Learned Inversion"], fix_range = True)
        plt.savefig(save_folder + "chest/image_{}.png".format(j))
        plt.close()

    try:
        os.mkdir(save_folder + "brain/")
    except:
        pass
    y_true = np.load("Datasets/brain_testing.npy").astype(np.float32)
    z = np.load("Datasets/brain_testing_recon.npy").astype(np.float32)
    x = np.load("Datasets/brain_testing_recon_limited.npy").astype(np.float32)
    d = np.load("Datasets/brain_testing_data_limited.npy").astype(np.float32)

    X = torch.from_numpy(d).to(dev)
    X = torch.unsqueeze(X,1)

    with torch.no_grad():
        Y_chest = net_chest(X)
        y_chest = np.squeeze(Y_chest.cpu().detach().numpy())
        Y_brain = net_brain(X)
        y_brain = np.squeeze(Y_brain.cpu().detach().numpy())
        Y_both = net_both(X)
        y_both = np.squeeze(Y_both.cpu().detach().numpy())

    rmse_full = np.mean((z- y_true)**2,axis = (1,2))**(1/2)
    rmse_limited = np.mean((x- y_true)**2,axis = (1,2))**(1/2)
    rmse_chest = np.mean((y_chest - y_true)**2, axis = (1,2))**(1/2)
    rmse_brain = np.mean((y_brain - y_true)**2, axis = (1,2))**(1/2)
    rmse_both = np.mean((y_both - y_true)**2, axis = (1,2))**(1/2)

    rmses = [rmse_full, rmse_limited, rmse_chest, rmse_brain, rmse_both]
    fig,ax = plt.subplots(1,1)
    ax.violinplot(rmses,
                  showmeans=False,
                  showmedians=True)
    
    ax.set_xticks([y + 1 for y in range(len(rmses))],
                  labels=['Full', 'Limited', 'Chest', 'Brain', 'Both'])
    plt.savefig(save_folder + "brain/hist.png")
    plt.close()
    for j in range(x.shape[0]):
        show2D([y_true[j,:,:],  z[j,:,:], x[j,:,:], y_chest[j,:,:], y_brain[j,:,:], y_both[j,:,:]], title = ["True", "Full Recon", "Limited Recon", "Chest Learned Inversion", "Brain Learned Inversion", "Both Learned Inversion"], fix_range = True)
        plt.savefig(save_folder + "brain/image_{}.png".format(j))
        plt.close()