
import numpy as np
from cil.framework import  AcquisitionGeometry, ImageData, AcquisitionData, ImageGeometry
from cil.plugins.astra.operators import ProjectionOperator
from cil.recon import FBP
import matplotlib.pyplot as plt

import os
import sys
sys.path.path.append('./')
from networks import *

from cil.utilities.display import show2D


def upscale_data(x):
    out = np.zeros((x.shape[0], 5*x.shape[1], x.shape[2])).astype(np.float32)
    for j in range(5):
        out[:,j::5,:]= x
    return out

if __name__ == "__main__":


    save_folder = "Data_Upscaler_Results/"
    try:
        os.mkdir(save_folder)
    except:
        pass
    dev = torch.device('cuda:0')

    net_brain = DataUNet().to(dev)
    net_brain.load_state_dict(torch.load('State_Dictionaries/brain_data_upscaler'))
    net_brain.eval()

    net_chest = DataUNet().to(dev)
    net_chest.load_state_dict(torch.load('State_Dictionaries/chest_data_upscaler'))
    net_chest.eval()

    net_both = DataUNet().to(dev)
    net_both.load_state_dict(torch.load('State_Dictionaries/both_data_upscaler'))
    net_both.eval()

    angles  = np.arange(180)
    na = len(angles)
    N = 256



    try:
        os.mkdir(save_folder + "chest/")
    except:
        pass

    Data = np.load("Datasets/chest_testing_data.npy").astype(np.float32)
    data_small = np.load("Datasets/chest_testing_data_limited.npy").astype(np.float32)
    Data_small = upscale_data(data_small)

    X = torch.from_numpy(Data_small).to(dev)
    X = torch.unsqueeze(X,1)

    with torch.no_grad():
        Y_chest = net_chest(X)
        data_chest = np.squeeze(Y_chest.cpu().detach().numpy())
        Y_brain = net_brain(X)
        data_brain = np.squeeze(Y_brain.cpu().detach().numpy())
        Y_both = net_both(X)
        data_both = np.squeeze(Y_both.cpu().detach().numpy())
    

    for j in range(Data.shape[0]):
        show2D([Data[j,:,:],  data_small[j,:,:], Data_small[j,:,:], data_chest[j,:,:], data_brain[j,:,:], data_both[j,:,:]], title = ["True Data", "Subsampled Data", "Copied Data", "Chest Upscaled Data", "Brain Upscaled Data", "Both Upscaled Data"], fix_range = True)
        plt.savefig(save_folder + "chest/sino_{}.png".format(j))
        plt.close()


    try:
        os.mkdir(save_folder + "brain/")
    except:
        pass

    Data = np.load("Datasets/brain_testing_data.npy").astype(np.float32)
    data_small = np.load("Datasets/brain_testing_data_limited.npy").astype(np.float32)
    Data_small = upscale_data(data_small)

    X = torch.from_numpy(Data_small).to(dev)
    X = torch.unsqueeze(X,1)

    with torch.no_grad():
        Y_chest = net_chest(X)
        data_chest = np.squeeze(Y_chest.cpu().detach().numpy())
        Y_brain = net_brain(X)
        data_brain = np.squeeze(Y_brain.cpu().detach().numpy())
        Y_both = net_both(X)
        data_both = np.squeeze(Y_both.cpu().detach().numpy())
    

    for j in range(Data.shape[0]):
        show2D([Data[j,:,:],  data_small[j,:,:], Data_small[j,:,:], data_chest[j,:,:], data_brain[j,:,:], data_both[j,:,:]], title = ["True Data", "Subsampled Data", "Copied Data", "Chest Upscaled Data", "Brain Upscaled Data", "Both Upscaled Data"], fix_range = True)
        plt.savefig(save_folder + "brain/sino_{}.png".format(j))
        plt.close()
