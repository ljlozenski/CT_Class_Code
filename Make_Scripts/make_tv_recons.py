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
    A = ProjectionOperator(ig, ag, device='gpu')
    ag_limited = AcquisitionGeometry.create_Parallel2D().set_angles(angles_limited).set_panel(N, pixel_size=1.0)
    A_limited = ProjectionOperator(ig, ag_limited, device='gpu')

    x_true = np.load("Datasets/chest_testing.npy")[0,:,:]
    #data = np.load("Datasets/chest_testing_data_limited.npy")[0,:,:]
    data = np.load("Datasets/chest_testing_data.npy")[0,:,:]
    snr = 0.05
    data += snr*np.mean(data**2)**(1/2)*np.random.standard_normal(data.shape)

    Data = ag_limited.allocate()
    Data.array = data

    
    #F = LeastSquares(A = A_limited, b = Data, c = 0.5)
    F = LeastSquares(A = A, b = Data, c = 0.5)
    gamma = 2e1
    G = gamma*TotalVariation()
    x0 = ig.allocate()
    fista = FISTA(f = F, g = G, initial = x0)
    fista.run(100)
    rec_fista = fista.solution

    rrmse = ((np.mean((x_true - rec_fista.array)**2))/np.mean(x_true**2))**(1/2)
    print(rrmse)

    fig = show2D([x_true, rec_fista.array], fix_range = (x_true.min(), x_true.max()))
    fig.save('recon_test.png')

