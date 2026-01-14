import numpy as np
import torch
import matplotlib.pyplot as plt

import os

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
        show2D([y_true[j,:,:],  z[j,:,:], x[j,:,:], y_chest[j,:,:], y_brain[j,:,:], y_both[j,:,:]], title = ["True", "Full Recon", "Limited Recon", "Chest Measurement Correction", "Brain Measurement Correction", "Both Measurement Correction"], fix_range = True)
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
        show2D([y_true[j,:,:],  z[j,:,:], x[j,:,:], y_chest[j,:,:], y_brain[j,:,:], y_both[j,:,:]], title = ["True", "Full Recon", "Limited Recon", "Chest Artifact Removal", "Brain Artifact Removal", "Both Artifact Removal"], fix_range = True)
        plt.savefig(save_folder + "brain/image_{}.png".format(j))
        plt.close()


    

