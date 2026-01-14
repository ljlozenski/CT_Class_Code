
import numpy as np
from cil.framework import  AcquisitionGeometry, ImageData, AcquisitionData, ImageGeometry
from cil.plugins.astra.operators import ProjectionOperator
from cil.recon import FBP

import os

import sys
sys.path.append('./')
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


    ag = AcquisitionGeometry.create_Parallel2D().set_angles(angles).set_panel(N, pixel_size=1.0)
    ig = ag.get_ImageGeometry()
    X = ig.allocate()
    A = ProjectionOperator(ig, ag, device='gpu')

    try:
        os.mkdir(save_folder + "chest/")
    except:
        pass

    Data = np.load("Datasets/chest_testing_data.npy").astype(np.float32)
    Data_small = np.load("Datasets/chest_testing_data_limited.npy").astype(np.float32)
    Data_small = upscale_data(Data_small)

    X = torch.from_numpy(Data_small).to(dev)
    X = torch.unsqueeze(X,1)

    with torch.no_grad():
        Y_chest = net_chest(X)
        data_chest = np.squeeze(Y_chest.cpu().detach().numpy())
        Y_brain = net_brain(X)
        data_brain = np.squeeze(Y_brain.cpu().detach().numpy())
        Y_both = net_both(X)
        data_both = np.squeeze(Y_both.cpu().detach().numpy())
    
    
    recon_chest = np.zeros((Data.shape[0], N,N))
    recon_brain = np.zeros((Data.shape[0], N,N))
    recon_both = np.zeros((Data.shape[0], N,N))

    data = ag.allocate()

    print("Starting Brain Chest Data")
    for j in range(Data.shape[0]):
        print(" Image {}/{}".format(j, Data.shape[0]))
        data.array = data_chest[j,:,:]
        recon = FBP(data, ig, backend='astra',filter='ram-lak').run(verbose=0)
        recon_chest[j,:,:] = recon.array

        data.array = data_brain[j,:,:]
        recon = FBP(data, ig, backend='astra',filter='ram-lak').run(verbose=0)
        recon_brain[j,:,:] = recon.array

        data.array = data_both[j,:,:]
        recon = FBP(data, ig, backend='astra',filter='ram-lak').run(verbose=0)
        recon_both[j,:,:] = recon.array
    np.save(save_folder + "chest/recon_chest", recon_chest)
    np.save(save_folder + "chest/recon_brain", recon_brain)
    np.save(save_folder + "chest/recon_both", recon_both)

    try:
        os.mkdir(save_folder + "brain/")
    except:
        pass

    Data = np.load("Datasets/brain_testing_data.npy").astype(np.float32)
    Data_small = np.load("Datasets/brain_testing_data_limited.npy").astype(np.float32)
    Data_small = upscale_data(Data_small)

    X = torch.from_numpy(Data_small).to(dev)
    X = torch.unsqueeze(X,1)

    with torch.no_grad():
        Y_chest = net_chest(X)
        data_chest = np.squeeze(Y_chest.cpu().detach().numpy())
        Y_brain = net_brain(X)
        data_brain = np.squeeze(Y_brain.cpu().detach().numpy())
        Y_both = net_both(X)
        data_both = np.squeeze(Y_both.cpu().detach().numpy())
    
    
    recon_chest = np.zeros((Data.shape[0], N,N))
    recon_brain = np.zeros((Data.shape[0], N,N))
    recon_both = np.zeros((Data.shape[0], N,N))

    data = ag.allocate()
    print("Starting Brain Data")
    for j in range(Data.shape[0]):
        print(" Image {}/{}".format(j, Data.shape[0]))
        data.array = data_chest[j,:,:]
        recon = FBP(data, ig, backend='astra',filter='ram-lak').run(verbose=0)
        recon_chest[j,:,:] = recon.array

        data.array = data_brain[j,:,:]
        recon = FBP(data, ig, backend='astra',filter='ram-lak').run(verbose=0)
        recon_brain[j,:,:] = recon.array

        data.array = data_both[j,:,:]
        recon = FBP(data, ig, backend='astra',filter='ram-lak').run(verbose=0)
        recon_both[j,:,:] = recon.array
    np.save(save_folder + "brain/recon_chest", recon_chest)
    np.save(save_folder + "brain/recon_brain", recon_brain)
    np.save(save_folder + "brain/recon_both", recon_both)





    