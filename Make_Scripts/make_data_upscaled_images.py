import numpy as np
import torch
import matplotlib.pyplot as plt

import os

from networks import *

from cil.utilities.display import show2D

if __name__ == "__main__":
    save_folder = "Data_Upscaler_Results/"
    try:
        os.mkdir(save_folder)
    except:
        pass

    try:
        os.mkdir(save_folder + "chest/")
    except:
        pass
    y_true = np.load("Datasets/chest_testing.npy").astype(np.float32)
    z = np.load("Datasets/chest_testing_recon.npy").astype(np.float32)

    x = np.load("Datasets/chest_testing_recon_limited.npy").astype(np.float32)


        
    y_chest = np.load(save_folder + "chest/recon_chest.npy")
    y_brain = np.load(save_folder + "chest/recon_brain.npy")
    y_both = np.load(save_folder + "chest/recon_both.npy")

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

    y_chest = np.load(save_folder + "brain/recon_chest.npy")
    y_brain = np.load(save_folder + "brain/recon_brain.npy")
    y_both = np.load(save_folder + "brain/recon_both.npy")

    for j in range(x.shape[0]):
        print(j)

        #fig,a = plt.subplots(1,3)
        #a[0].imshow(y_true[j,:,:], vmin = 0, vmax = 1, cmap = 'gray')
        #a[1].imshow(x[j,:,:], vmin = 0, vmax = 1, cmap = 'gray')
        #a[2].imshow(y[j,:,:], vmin = 0, vmax = 1, cmap = 'gray')

        show2D([y_true[j,:,:],  z[j,:,:], x[j,:,:], y_chest[j,:,:], y_brain[j,:,:], y_both[j,:,:]], title = ["True", "Full Recon", "Limited Recon", "Chest Artifact Removal", "Brain Artifact Removal", "Both Artifact Removal"], fix_range = True)
        plt.savefig(save_folder + "brain/image_{}.png".format(j))
        plt.close()


    

