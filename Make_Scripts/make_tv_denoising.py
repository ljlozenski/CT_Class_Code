import numpy as np
from cil.framework import  AcquisitionGeometry, ImageData, AcquisitionData, ImageGeometry
from cil.plugins.astra.operators import ProjectionOperator
from cil.recon import FBP
from cil.optimisation.functions import L2NormSquared, TotalVariation, LeastSquares
from cil.optimisation.algorithms import CGLS, FISTA

from cil.utilities.display import show2D


if __name__ == "__main__":
    angles  = np.arange(180)
    angles_limited = angles[::5]
    N = 256

    ag = AcquisitionGeometry.create_Parallel2D().set_angles(angles).set_panel(N, pixel_size=1.0)
    ig = ag.get_ImageGeometry()
    

    x_true = np.load("Datasets/chest_testing.npy")[0,:,:]
    snr = 0.05
    std = np.mean(x_true**2)**(1/2)
    x_noisy = x_true + snr*std*np.random.standard_normal(x_true.shape)

    
    gamma = 2e1
    G = gamma*TotalVariation()
    
    x0 = ig.allocate()
    x0.array = x_noisy
    denoise = G.proximal(x0)

    rrmse = ((np.mean((x_true - denoise)**2))/np.mean(x_true**2))**(1/2)
    print(rrmse)

    """fig = show2D([x_true, rec_fista.array], fix_range = (x_true.min(), x_true.max()))
    fig.save('recon_test.png')""""

