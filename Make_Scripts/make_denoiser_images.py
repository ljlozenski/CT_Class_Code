import numpy as np
import torch
import matplotlib.pyplot as plt

import os
import sys
sys.path.append("./")
from networks import * 


from cil.utilities.display import show2D
from cil.optimisation.functions import TotalVariation
from cil.framework import  AcquisitionGeometry


if __name__ == "__main__":
    save_folder = "Denoiser_Results/"
    try:
        os.mkdir(save_folder)
    except:
        pass
    dev = torch.device('cuda:0')


    net_chest = Denoiser().to(dev)
    net_chest.load_state_dict(torch.load('State_Dictionaries/chest_denoiser'))
    net_chest.eval()

    net_brain = Denoiser().to(dev)
    net_brain.load_state_dict(torch.load('State_Dictionaries/brain_denoiser'))
    net_brain.eval()

    net_both = Denoiser().to(dev)
    net_both.load_state_dict(torch.load('State_Dictionaries/both_denoiser'))
    net_both.eval()



    #noise_levels =  [0.001, 0.01, 0.1, 0.5, 1]
    #noise_levels = [0.001, 0.01, 0.05, 0.1, 0.2]
    noise_level = 0.2
    
    try:
        os.mkdir(save_folder + "chest/")
    except:
        pass
    x = np.load("Datasets/chest_testing.npy").astype(np.float32)
    std = np.mean(x**2)**(1/2)
    x_noisy = x + std*noise_level*np.random.standard_normal(x.shape)

    y_tv = 0*x_noisy
    gamma = 1e-1
    G = gamma*TotalVariation()
    angles  = np.arange(180)
    angles_limited = angles[::5]
    N = 256
    

    ag = AcquisitionGeometry.create_Parallel2D().set_angles(angles).set_panel(N, pixel_size=1.0)
    ig = ag.get_ImageGeometry()
    x_cil = ig.allocate()

    for j in range(x_noisy.shape[0]):
        print(j)
        x_cil.array = x_noisy[j,:,:]
        y_tv[j,:,:] = G.proximal(x_cil, 1.).array

    X = torch.from_numpy(x_noisy).float().to(dev)
    X = torch.unsqueeze(X,1)


    with torch.no_grad():
        Y_chest = net_chest(X)
        y_chest = np.squeeze(Y_chest.cpu().detach().numpy())
        Y_brain = net_brain(X)
        y_brain = np.squeeze(Y_brain.cpu().detach().numpy())
        Y_both = net_both(X)
        y_both = np.squeeze(Y_both.cpu().detach().numpy())
    
    
    
    rmse_noise = np.mean((x_noisy - x)**2, axis = (1,2))**(1/2)
    rmse_tv = np.mean((y_tv - x)**2, axis = (1,2))**(1/2)
    rmse_chest = np.mean((y_chest - x)**2, axis = (1,2))**(1/2)
    rmse_brain = np.mean((y_brain - x)**2, axis = (1,2))**(1/2)
    rmse_both = np.mean((y_both - x)**2, axis = (1,2))**(1/2)

    
    rmses = [rmse_noise, rmse_tv, rmse_chest, rmse_brain, rmse_both]
    fig,ax = plt.subplots(1,1)
    ax.violinplot(rmses,
                  showmeans=False,
                  showmedians=True)
    
    ax.set_xticks([y + 1 for y in range(len(rmses))],
                  labels=['Noisy', 'TV', 'Chest', 'Brain', 'Both'])
    plt.savefig(save_folder + "chest/hist.png")
    plt.close()

    

            

    for j in range(x.shape[0]):
        print(j)

        #fig,a = plt.subplots(1,3)
        #a[0].imshow(y_true[j,:,:], vmin = 0, vmax = 1, cmap = 'gray')
        #a[1].imshow(x[j,:,:], vmin = 0, vmax = 1, cmap = 'gray')
        #a[2].imshow(y[j,:,:], vmin = 0, vmax = 1, cmap = 'gray')

        show2D([x[j,:,:],  x_noisy[j,:,:], y_tv[j,:,:], y_chest[j,:,:], y_brain[j,:,:], y_both[j,:,:]], title = ["True", "Noisy", "TV Denoiser", "Chest Denoiser", "Brain Denoiser", "Both Denoiser"], fix_range = True)
        plt.savefig(save_folder + "chest/image_{}.png".format(j))
        plt.close()


    try:
        os.mkdir(save_folder + "brain/")
    except:
        pass
    x = np.load("Datasets/brain_testing.npy").astype(np.float32)
    std = np.mean(x**2)**(1/2)
    x_noisy = x + std*noise_level*np.random.standard_normal(x.shape)

    y_tv = 0*x_noisy
    gamma = 1e-1
    G = gamma*TotalVariation()
    angles  = np.arange(180)
    angles_limited = angles[::5]
    N = 256
    

    ag = AcquisitionGeometry.create_Parallel2D().set_angles(angles).set_panel(N, pixel_size=1.0)
    ig = ag.get_ImageGeometry()
    x_cil = ig.allocate()

    for j in range(x_noisy.shape[0]):
        print(j)
        x_cil.array = x_noisy[j,:,:]
        y_tv[j,:,:] =  G.proximal(x_cil, 1.).array

    X = torch.from_numpy(x_noisy).float().to(dev)
    X = torch.unsqueeze(X,1)


    with torch.no_grad():
        Y_chest = net_chest(X)
        y_chest = np.squeeze(Y_chest.cpu().detach().numpy())
        Y_brain = net_brain(X)
        y_brain = np.squeeze(Y_brain.cpu().detach().numpy())
        Y_both = net_both(X)
        y_both = np.squeeze(Y_both.cpu().detach().numpy())



    rmse_tv = np.mean((y_tv - x)**2, axis = (1,2))**(1/2)
    rmse_chest = np.mean((y_chest - x)**2, axis = (1,2))**(1/2)
    rmse_brain = np.mean((y_brain - x)**2, axis = (1,2))**(1/2)
    rmse_both = np.mean((y_both - x)**2, axis = (1,2))**(1/2)

    
    rmses = [rmse_noise, rmse_tv, rmse_chest, rmse_brain, rmse_both]
    fig,ax = plt.subplots(1,1)
    ax.violinplot(rmses,
                  showmeans=False,
                  showmedians=True)
    ax.set_xticks([y + 1 for y in range(len(rmses))],
                  labels=['Noisy', 'TV', 'Chest', 'Brain', 'Both'])
    plt.savefig(save_folder + "brain/hist.png")
    plt.close()

    for j in range(x.shape[0]):
        print(j)

        #fig,a = plt.subplots(1,3)
        #a[0].imshow(y_true[j,:,:], vmin = 0, vmax = 1, cmap = 'gray')
        #a[1].imshow(x[j,:,:], vmin = 0, vmax = 1, cmap = 'gray')
        #a[2].imshow(y[j,:,:], vmin = 0, vmax = 1, cmap = 'gray')

        show2D([x[j,:,:],  x_noisy[j,:,:], y_tv[j,:,:], y_chest[j,:,:], y_brain[j,:,:], y_both[j,:,:]], title = ["True", "Noisy", "TV Denoiser", "Chest Denoiser", "Brain Denoiser", "Both Denoiser"], fix_range = True)
        plt.savefig(save_folder + "brain/image_{}.png".format(j))
        plt.close()
        rmse_noise = np.mean((x_noisy - x)**2, axis = (1,2))**(1/2)




