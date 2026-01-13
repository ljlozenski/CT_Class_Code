import numpy as np
import torch
import matplotlib.pyplot as plt

import os

from cil.utilities.display import show2D

if __name__ == "__main__":
    save_folder = "PNP_Results/"
    try:
        os.mkdir(save_folder)
    except:
        pass

    try:
        os.mkdir(save_folder + "chest/")
    except:
        pass
    y_true = np.load("Datasets/chest_testing.npy").astype(np.float32)

    y_fbp = np.load(save_folder + "chest/recon_fbp.npy")
    y_tv = np.load(save_folder + "chest/recon_tv.npy")
    y_chest = np.load(save_folder + "chest/recon_chest.npy")
    y_brain = np.load(save_folder + "chest/recon_brain.npy")
    y_both = np.load(save_folder + "chest/recon_both.npy")

    rmse_fbp = np.mean((y_fbp - y_true)**2, axis = (1,2))**(1/2)
    rmse_tv = np.mean((y_tv - y_true)**2, axis = (1,2))**(1/2)
    rmse_chest = np.mean((y_chest - y_true)**2, axis = (1,2))**(1/2)
    rmse_brain = np.mean((y_brain - y_true)**2, axis = (1,2))**(1/2)
    rmse_both = np.mean((y_both - y_true)**2, axis = (1,2))**(1/2)

    
    rmses = [rmse_fbp, rmse_tv, rmse_chest, rmse_brain, rmse_both]
    fig,ax = plt.subplots(1,1)
    ax.violinplot(rmses,
                  showmeans=False,
                  showmedians=True)
    
    ax.set_xticks([y + 1 for y in range(len(rmses))],
                  labels=['FBP', 'TV', 'Chest', 'Brain', 'Both'])
    plt.savefig(save_folder + "chest/hist.png")
    plt.close()

    """for j in range(y_true.shape[0]):
        show2D([y_true[j,:,:],  y_fbp[j,:,:], y_tv[j,:,:], y_chest[j,:,:], y_brain[j,:,:], y_both[j,:,:]], title = ["True", "FBP", "TV", "Chest PNP", "Brain PNP", "Both PNP"], fix_range = True)
        plt.savefig(save_folder + "chest/image_{}.png".format(j))
        plt.close()"""



    try:
        os.mkdir(save_folder + "brain/")
    except:
        pass
    y_true = np.load("Datasets/brain_testing.npy").astype(np.float32)
    y_fbp = np.load(save_folder + "brain/recon_fbp.npy")
    y_tv = np.load(save_folder + "brain/recon_tv.npy")
    y_chest = np.load(save_folder + "brain/recon_chest.npy")
    y_brain = np.load(save_folder + "brain/recon_brain.npy")
    y_both = np.load(save_folder + "brain/recon_both.npy")


    rmse_fbp = np.mean((y_fbp - y_true)**2, axis = (1,2))**(1/2)
    rmse_tv = np.mean((y_tv - y_true)**2, axis = (1,2))**(1/2)
    rmse_chest = np.mean((y_chest - y_true)**2, axis = (1,2))**(1/2)
    rmse_brain = np.mean((y_brain - y_true)**2, axis = (1,2))**(1/2)
    rmse_both = np.mean((y_both - y_true)**2, axis = (1,2))**(1/2)

    
    rmses = [rmse_fbp, rmse_tv, rmse_chest, rmse_brain, rmse_both]
    fig,ax = plt.subplots(1,1)
    ax.violinplot(rmses,
                  showmeans=False,
                  showmedians=True)
    
    ax.set_xticks([y + 1 for y in range(len(rmses))],
                  labels=['FBP', 'TV', 'Chest', 'Brain', 'Both'])
    plt.savefig(save_folder + "brain/hist.png")
    plt.close()

    """for j in range(y_true.shape[0]):
        show2D([y_true[j,:,:],  y_fbp[j,:,:], y_tv[j,:,:], y_chest[j,:,:], y_brain[j,:,:], y_both[j,:,:]], title = ["True", "FBP", "TV", "Chest PNP", "Brain PNP", "Both PNP"], fix_range = True)
        plt.savefig(save_folder + "brain/image_{}.png".format(j))
        plt.close()"""