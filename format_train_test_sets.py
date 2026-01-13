import os
import numpy as np
import matplotlib.pyplot as plt


if __name__ == "__main__":

    brain_train_path = 'Datasets/brain_data/Dataset/images/trainA/'
    brain_train_files = next(os.walk(brain_train_path), (None, None, []))[2]
    nbrain_train = len(brain_train_files)
    brain_training_set = np.zeros((nbrain_train, 256,256))

    for j in range(nbrain_train):
        file = brain_train_files[j]
        brain = plt.imread(brain_train_path + file)[::2,::2]
        brain_training_set[j,:,:] = brain
    np.save('Datasets/brain_training.npy', brain_training_set)
    print("Saved {} Training Examples of Brain CT Images".format(nbrain_train))

    brain_test_path = 'Datasets/brain_data/Dataset/images/testA/'
    brain_test_files = next(os.walk(brain_test_path), (None, None, []))[2]
    nbrain_test = len(brain_test_files)
    brain_testing_set = np.zeros((nbrain_test, 256,256))

    for j in range(nbrain_test):
        file = brain_test_files[j]
        brain = plt.imread(brain_test_path + file)[::2,::2]
        brain_testing_set[j,:,:] = brain
    np.save('Datasets/brain_testing.npy', brain_testing_set)
    print("Saved {} Testing Examples of Brain CT Images".format(nbrain_test))

    chest_train_parent = 'Datasets/chest_data/Data/train/'
    chest_train_paths = ['adenocarcinoma_left.lower.lobe_T2_N0_M0_Ib/', 'large.cell.carcinoma_left.hilum_T2_N2_M0_IIIa/', 'squamous.cell.carcinoma_left.hilum_T1_N2_M0_IIIa/']
    Chest_training_set = np.zeros((0, 256,256))
    for chest_train_path in chest_train_paths:
        chest_train_files = next(os.walk(chest_train_parent + chest_train_path), (None, None, []))[2]
        nchest_train = len(chest_train_files)
        print(nchest_train)
        chest_training_set = np.zeros((nchest_train, 256,256))
        for j in range(nchest_train):
            file = chest_train_files[j]
            try:
                chest = plt.imread(chest_train_parent + chest_train_path + file)[:,:,0]
            except:
                chest = plt.imread(chest_train_parent + chest_train_path + file)
            ny = chest.shape[1]
            m = int(np.ceil(ny/256))
            chest = chest[m-1::m, m-1::m]
            #print(chest.shape)
            nx, ny = chest.shape
            #print(nx,ny, chest_train_parent + chest_train_path + file)
            dx0 = int(np.floor((256-nx)/2))
            dx1 = int(np.ceil((256-nx)/2))
            dy0 = int(np.floor((256-ny)/2))
            dy1 = int(np.ceil((256-ny)/2))
            chest_training_set[j,:,:]= np.pad(chest, ((dx0, dx1), (dy0,dy1)))
        Chest_training_set = np.concatenate((Chest_training_set, chest_training_set), axis = 0)
    nchest_train = Chest_training_set.shape[0]
    np.save('Datasets/chest_training.npy',Chest_training_set)
    print("Saved {} Training Examples of Chest CT Images".format(nchest_train))

    chest_test_parent = 'Datasets/chest_data/Data/test/'
    chest_test_paths = ['adenocarcinoma/', 'large.cell.carcinoma/', 'squamous.cell.carcinoma/']
    Chest_testing_set = np.zeros((0, 256,256))
    for chest_test_path in chest_test_paths:
        chest_test_files = next(os.walk(chest_test_parent + chest_test_path), (None, None, []))[2]
        nchest_test = len(chest_test_files)
        chest_testing_set = np.zeros((nchest_test, 256,256))
        for j in range(nchest_test):
            file = chest_test_files[j]
            try:
                chest = plt.imread(chest_test_parent + chest_test_path + file)[:,:,0]
            except:
                chest = plt.imread(chest_test_parent + chest_test_path + file)
            ny = chest.shape[1]
            m = int(np.ceil(ny/256))
            chest = chest[m-1::m, m-1::m]
            #print(chest.shape)
            nx, ny = chest.shape
            #print(nx,ny, chest_test_parent + chest_test_path + file)
            dx0 = int(np.floor((256-nx)/2))
            dx1 = int(np.ceil((256-nx)/2))
            dy0 = int(np.floor((256-ny)/2))
            dy1 = int(np.ceil((256-ny)/2))
            chest_testing_set[j,:,:]= np.pad(chest, ((dx0, dx1), (dy0,dy1)))
        Chest_testing_set = np.concatenate((Chest_testing_set, chest_testing_set), axis = 0)
    nchest_test = Chest_testing_set.shape[0]
    np.save('Datasets/chest_testing.npy',Chest_testing_set)
    print("Saved {} testing Examples of Chest CT Images".format(nchest_test))



    

