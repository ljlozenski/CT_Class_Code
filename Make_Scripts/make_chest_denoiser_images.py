import numpy as np
import torch
import matplotlib.pyplot as plt

import os

from networks import *

from cil.utilities.display import show2D

if __name__ == "__main__":
    save_folder = "Chest_Results/"
    try:
        os.mkdir(save_folder)
    except:
        pass
    dev = torch.device('cuda:0')


    net_chest = ImageUNet().to(dev)
    net_chest.load_state_dict(torch.load('State_Dictionaries/chest_denoiser'))
    net_chest.eval()

    #noise_levels =  [0.001, 0.01, 0.1, 0.5, 1]
    noise_levels = [0.001, 0.01, 0.05, 0.1, 0.2]
    

    x = np.load("Datasets/chest_testing.npy").astype(np.float32)
    std = np.mean(x**2)**(1/2)



    YS = []

    with torch.no_grad():
        for nl in noise_levels:
            xn = x + std*nl*np.random.standard_normal(x.shape)
            X = torch.from_numpy(xn).float().to(dev)
            X = torch.unsqueeze(X,1)
            Y = torch.squeeze(X + 0.1*0.25*net_chest(X))
            YS.append(Y.cpu().detach().numpy())
            
    titles = ["True"] + [str(nl) for nl in noise_levels]
    for j in range(x.shape[0]):
        print(j)

        #fig,a = plt.subplots(1,3)
        #a[0].imshow(y_true[j,:,:], vmin = 0, vmax = 1, cmap = 'gray')
        #a[1].imshow(x[j,:,:], vmin = 0, vmax = 1, cmap = 'gray')
        #a[2].imshow(y[j,:,:], vmin = 0, vmax = 1, cmap = 'gray')
        Ys = [x[j,:,:]] + [Y[j,:,:] for Y in YS]
        

        show2D(Ys, title = titles, fix_range = True)
        plt.savefig(save_folder + "image_{}.png".format(j))
        plt.close()


