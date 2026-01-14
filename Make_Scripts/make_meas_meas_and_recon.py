
import numpy as np
from cil.framework import  AcquisitionGeometry, ImageData, AcquisitionData, ImageGeometry
from cil.plugins.astra.operators import ProjectionOperator
from cil.recon import FBP

if __name__ == "__main__":
    parent_folder = "Datasets/"
    input_labels = ["brain_training", "brain_testing", "chest_training", "chest_testing"]

    angles  = np.arange(180)
    angles_limited = angles[::5]
    na = len(angles)
    na_limited = len(angles_limited)
    N = 256


    ag = AcquisitionGeometry.create_Parallel2D().set_angles(angles).set_panel(N, pixel_size=1.0)
    ig = ag.get_ImageGeometry()
    X = ig.allocate()
    ag_limited = AcquisitionGeometry.create_Parallel2D().set_angles(angles_limited).set_panel(N, pixel_size=1.0)
    A = ProjectionOperator(ig, ag, device='gpu')
    A_limited = ProjectionOperator(ig, ag_limited, device='gpu')


    for label in input_labels:
        print("Starting reconstructions on ", label)
        gt = np.load(parent_folder + label + ".npy")

        Recon = np.zeros(gt.shape)
        Recon_limited = np.zeros(gt.shape)

        Data = np.zeros((gt.shape[0], na, N))
        Data_limited = np.zeros((gt.shape[0], na_limited, N))

        for j in range(gt.shape[0]):
            print(" Image {}/{}".format(j, gt.shape[0]))
            X.array = gt[j,:,:]
            data = A.direct(X)
            Data[j,:,:] = data.array
            recon = FBP(data, ig, backend='astra',filter='ram-lak').run(verbose=0)
            Recon[j,:,:] = recon.array

            data_limited = A_limited.direct(X)
            Data_limited[j,:,:] = data_limited.array
            recon = FBP(data_limited, ig, backend='astra',filter='ram-lak').run(verbose=0)
            Recon_limited[j,:,:] = recon.array

        np.save(parent_folder + label + "_recon", Recon)
        np.save(parent_folder + label + "_data", Data)
        np.save(parent_folder + label + "_recon_limited", Recon_limited)
        np.save(parent_folder + label + "_data_limited", Data_limited)



    