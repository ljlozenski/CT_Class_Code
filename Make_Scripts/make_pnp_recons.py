import numpy as np
from cil.framework import  AcquisitionGeometry
from cil.plugins.astra.operators import ProjectionOperator
from cil.recon import FBP
from cil.optimisation.functions import TotalVariation, LeastSquares, Function
from cil.optimisation.algorithms import FISTA, CGLS

from cil.utilities.display import show2D
import torch

import sys
sys.path.append("./")
from networks import * 
import os





class PNP_Function(Function):
    def __init__(self, Net, c= 1):
        self.net = Net
        self.c = c
        self.dev = next(Net.parameters()).get_device()
    def __call__(self, x):
        return 0
    def proximal(self,x, tau, out = None):
        X = torch.from_numpy(x.array).float().to(self.dev)
        X = torch.unsqueeze(X,0)
        X = torch.unsqueeze(X,0)
        Xd = torch.squeeze(self.net(X))
        xd = Xd.cpu().detach().numpy()
        if out is None:
            out = x*0
        gamma = self.c*tau
        gamma = gamma/(1+ gamma)
        out.fill(gamma*xd)
        out += (1-gamma)*x
        return out
    


if __name__ == "__main__":
    save_folder = "PNP_Results/"
    try:
        os.mkdir(save_folder)
    except:
        pass

    angles  = np.arange(180)
    N = 256

    ag = AcquisitionGeometry.create_Parallel2D().set_angles(angles).set_panel(N, pixel_size=1.0)
    ig = ag.get_ImageGeometry()
    A = ProjectionOperator(ig, ag, device='gpu')
    
    


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


    gamma = 5e3
    G_chest = PNP_Function(net_chest, c = gamma)
    G_brain = PNP_Function(net_brain, c = gamma)
    G_both = PNP_Function(net_both, c = gamma)

    beta = 1e2
    G_Tv = beta*TotalVariation()

    x0 = ig.allocate()
    data = ag.allocate()

    
    snr = 0.05
    std = snr*40

    try:
        os.mkdir(save_folder + "chest/")
    except:
        pass
    Data = np.load("Datasets/chest_testing_data.npy")
    Data += std*np.random.standard_normal(Data.shape)
    Data = Data.astype(np.float32)

    recon_chest = np.zeros((Data.shape[0], N,N))
    recon_brain = np.zeros((Data.shape[0], N,N))
    recon_both = np.zeros((Data.shape[0], N,N))
    #recon_tv = np.zeros((Data.shape[0], N,N))
    #recon_fbp = np.zeros((Data.shape[0], N,N))

    for j in range(Data.shape[0]):
        print(j)

        data.array = Data[j,:,:]
        F = LeastSquares(A = A, b = data, c = 0.5)

        fista_chest = FISTA(f = F, g = G_chest, initial = x0)
        fista_chest.run(100)
        recon_chest[j,:,:] = fista_chest.solution.array

        fista_brain = FISTA(f = F, g = G_brain, initial = x0)
        fista_brain.run(100)
        recon_brain[j,:,:] = fista_brain.solution.array

        fista_both = FISTA(f = F, g = G_both, initial = x0)
        fista_both.run(100)
        recon_both[j,:,:] = fista_both.solution.array
    
        #fista_tv = FISTA(f = F, g = G_Tv, initial = x0)
        #fista_tv.run(100)
        #recon_tv[j,:,:] = fista_tv.solution.array
        #recon_fbp[j,:,:] = FBP(data, ig, backend='astra',filter='ram-lak').run(verbose=0).array

    np.save(save_folder + "chest/recon_chest", recon_chest)
    np.save(save_folder + "chest/recon_brain", recon_brain)
    np.save(save_folder + "chest/recon_both", recon_both)
    #np.save(save_folder + "chest/recon_tv", recon_tv)
    #np.save(save_folder + "chest/recon_fbp", recon_fbp)

    try:
        os.mkdir(save_folder + "brain/")
    except:
        pass
    Data = np.load("Datasets/brain_testing_data.npy")
    Data += std*np.random.standard_normal(Data.shape)
    Data = Data.astype(np.float32)

    recon_chest = np.zeros((Data.shape[0], N,N))
    recon_brain = np.zeros((Data.shape[0], N,N))
    recon_both = np.zeros((Data.shape[0], N,N))
    #recon_tv = np.zeros((Data.shape[0], N,N))
    #recon_fbp = np.zeros((Data.shape[0], N,N))

    for j in range(Data.shape[0]):
        print(j)

        data.array = Data[j,:,:]
        F = LeastSquares(A = A, b = data, c = 0.5)

        fista_chest = FISTA(f = F, g = G_chest, initial = x0)
        fista_chest.run(100)
        recon_chest[j,:,:] = fista_chest.solution.array

        fista_brain = FISTA(f = F, g = G_brain, initial = x0)
        fista_brain.run(100)
        recon_brain[j,:,:] = fista_brain.solution.array

        fista_both = FISTA(f = F, g = G_both, initial = x0)
        fista_both.run(100)
        recon_both[j,:,:] = fista_both.solution.array

        #fista_tv = FISTA(f = F, g = G_Tv, initial = x0)
        #fista_tv.run(100)
        #recon_tv[j,:,:] = fista_tv.solution.array

        #recon_fbp[j,:,:] = FBP(data, ig, backend='astra',filter='ram-lak').run(verbose=0).array

    np.save(save_folder + "brain/recon_chest", recon_chest)
    np.save(save_folder + "brain/recon_brain", recon_brain)
    np.save(save_folder + "brain/recon_both", recon_both)
    #np.save(save_folder + "brain/recon_tv", recon_tv)
    #np.save(save_folder + "brain/recon_fbp", recon_fbp)


