import numpy as np
from cil.framework import  AcquisitionGeometry, ImageData, AcquisitionData, ImageGeometry
from cil.plugins.astra.operators import ProjectionOperator
from cil.utilities.display import show2D
from cil.plugins import TomoPhantom as cilTomoPhantom

import torch

from cil.plugins import TomoPhantom as cilTomoPhantom
from cil.utilities.display import show2D

class nf(torch.nn.Module):
    def __init__(self,width = 100, depth = 4):
        super().__init__()
        self.input = torch.nn.Linear(2,width)
        self.layers = torch.nn.ModuleList([torch.nn.Linear(width, width) for d in range(depth-1)])
        self.act = torch.nn.SiLU()
        self.output = torch.nn.Linear(width,1)
    def forward(self,x):
        x = self.act(self.input(x))
        for l in self.layers:
            x = self.act(l(x))
        x = self.output(x)
        return x
        


if __name__ == "__main__":
    n = 201
    ig = ImageGeometry(voxel_num_x=n, 
                    voxel_num_y=n, 
                    voxel_size_x=1, 
                    voxel_size_y=1)
    P = cilTomoPhantom.get_ImageData(num_model=1, geometry=ig)
    P = P.array
    P = P/np.mean(P**2)**(1/2)
    dev = torch.device('cuda:0')
    Pt = torch.from_numpy(P.flatten()).float().to(dev)

    grid_mks = np.arange(n) + 1/2 - n/2
    c0, c1 = np.meshgrid(grid_mks, grid_mks)
    xy = torch.from_numpy(np.stack((c0.flatten(), c1.flatten()), 1)).float().to(dev)

    net = nf().to(dev)
    nits = 10**4
    opt = torch.optim.Adam(net.parameters(), lr = 1e-3)

    for it in range(nits):
        opt.zero_grad()

        out = net(xy)[:,0]
        loss = torch.mean((out - Pt)**2)
        loss.backward()
        opt.step()
        print("Iteration ", it, " RRMSE: " loss.item()**(1/2))
    O = net(xy)[:,0].cpu().detach().numpy()
    O = O.reshape((n,n))

    E = P - O

    P_freq = np.fft.fft2(P)
    P_freq = np.fft.fftshift(P_freq, axes = (0,1))
    P_freq = np.log(np.abs(P_freq[n//2:,n//2:]))

    O_freq = np.fft.fft2(O)
    O_freq = np.fft.fftshift(O_freq, axes = (0,1))
    O_freq = np.log(np.abs(O_freq[n//2:,n//2:]))

    E_freq = np.fft.fft2(E)
    E_freq = np.fft.fftshift(E_freq, axes = (0,1))
    E_freq = np.log(np.abs(E_freq[n//2:,n//2:]))

    space_plot = show2D([P,O,E], fix_range = True)
    space_plot.save("Figures/space_plot.png")

    freq_plot = show2D([P_freq,O_freq,E_freq], fix_range = True)
    freq_plot.save("Figures/freq_plot.png")




