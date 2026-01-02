import numpy as np
import torch
import matplotlib.pyplot as plt

import os

from networks import *

from cil.utilities.display import show2D

if __name__ == "__main__":
    save_folder = "Artifact_Removal_Results/"
    try:
        os.mkdir(save_folder)
    except:
        pass
    dev = torch.device('cuda:0')
    net_brain = ImageUNet().to(dev)
    net_brain.load_state_dict(torch.load('State_Dictionaries/brain_artifact_remover'))
    net_brain.eval()

    net_chest = ImageUNet().to(dev)
    net_chest.load_state_dict(torch.load('State_Dictionaries/chest_artifact_remover'))
    net_chest.eval()

    net_both = ImageUNet().to(dev)
    net_both.load_state_dict(torch.load('State_Dictionaries/both_artifact_remover'))
    net_both.eval()

    try:
        os.mkdir(save_folder + "chest/")
    except:
        pass
    y_true = np.load("Datasets/chest_testing.npy").astype(np.float32)
    z = np.load("Datasets/chest_testing_recon.npy").astype(np.float32)

    x = np.load("Datasets/chest_testing_recon_limited.npy").astype(np.float32)

    X = torch.from_numpy(x).to(dev)
    X = torch.unsqueeze(X,1)

    with torch.no_grad():
        Y_chest = net_chest(X)
        y_chest = np.squeeze(Y_chest.cpu().detach().numpy())
        Y_brain = net_brain(X)
        y_brain = np.squeeze(Y_brain.cpu().detach().numpy())
        Y_both = net_both(X)
        y_both = np.squeeze(Y_both.cpu().detach().numpy())

    for j in range(x.shape[0]):
        print(j)

        #fig,a = plt.subplots(1,3)
        #a[0].imshow(y_true[j,:,:], vmin = 0, vmax = 1, cmap = 'gray')
        #a[1].imshow(x[j,:,:], vmin = 0, vmax = 1, cmap = 'gray')
        #a[2].imshow(y[j,:,:], vmin = 0, vmax = 1, cmap = 'gray')

        show2D([y_true[j,:,:],  z[j,:,:], x[j,:,:], y_chest[j,:,:], y_brain[j,:,:], y_both[j,:,:]], title = ["True", "Full Recon", "Limited Recon", "Chest Artifact Removal", "Brain Artifact Removal", "Both Artifact Removal"], fix_range = True)
        plt.savefig(save_folder + "chest/image_{}.png".format(j))
        plt.close()



    try:
        os.mkdir(save_folder + "brain/")
    except:
        pass
    y_true = np.load("Datasets/brain_testing.npy").astype(np.float32)
    z = np.load("Datasets/brain_testing_recon.npy").astype(np.float32)
    x = np.load("Datasets/brain_testing_recon_limited.npy").astype(np.float32)

    X = torch.from_numpy(x).to(dev)
    X = torch.unsqueeze(X,1)

    with torch.no_grad():
        Y_chest = net_chest(X)
        y_chest = np.squeeze(Y_chest.cpu().detach().numpy())
        Y_brain = net_brain(X)
        y_brain = np.squeeze(Y_brain.cpu().detach().numpy())
        Y_both = net_both(X)
        y_both = np.squeeze(Y_both.cpu().detach().numpy())
    for j in range(x.shape[0]):
        print(j)

        #fig,a = plt.subplots(1,3)
        #a[0].imshow(y_true[j,:,:], vmin = 0, vmax = 1, cmap = 'gray')
        #a[1].imshow(x[j,:,:], vmin = 0, vmax = 1, cmap = 'gray')
        #a[2].imshow(y[j,:,:], vmin = 0, vmax = 1, cmap = 'gray')

        show2D([y_true[j,:,:],  z[j,:,:], x[j,:,:], y_chest[j,:,:], y_brain[j,:,:], y_both[j,:,:]], title = ["True", "Full Recon", "Limited Recon", "Chest Artifact Removal", "Brain Artifact Removal", "Both Artifact Removal"], fix_range = True)
        plt.savefig(save_folder + "brain/image_{}.png".format(j))
        plt.close()


    

