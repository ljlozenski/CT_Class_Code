import numpy as np
import torch
import matplotlib.pyplot as plt

import sys
sys.path.append("./")
from networks import *

if __name__ == "__main__":

    dev = torch.device('cuda:0')

    x_train = np.load("Datasets/chest_training_recon_limited.npy").astype(np.float32)
    x_test = np.load("Datasets/chest_testing_recon_limited.npy").astype(np.float32)
    y_train = np.load("Datasets/chest_training.npy").astype(np.float32)
    y_test = np.load("Datasets/chest_testing.npy").astype(np.float32)

    X_train = torch.unsqueeze(torch.from_numpy(x_train),1)
    X_test = torch.unsqueeze(torch.from_numpy(x_test),1)
    Y_train = torch.unsqueeze(torch.from_numpy(y_train),1)
    Y_test = torch.unsqueeze(torch.from_numpy(y_test),1)

    net = ImageUNet().to(dev)

    num_batches = 20
    num_epochs = 10**3

    optimizer = torch.optim.Adam(net.parameters(), lr = 1e-4)


    train_losses = []
    test_losses = []

    inds = np.arange(x_train.shape[0])



    for epoch in range(num_epochs):
        np.random.shuffle(inds)
        
        net.train()
        train_loss = 0

        for b in range(num_batches):
            optimizer.zero_grad()
            X = X_train[inds[b::num_batches],:,:,:].to(dev)
            Y = Y_train[inds[b::num_batches],:,:,:].to(dev)

            out = net(X)

            loss = torch.mean((out - Y)**2)
            train_loss += loss.item()/num_batches
            loss.backward()
            optimizer.step()
        train_losses.append(train_loss)
        print("Epoch {}".format(epoch))
        print(" Training Loss: {}".format(train_loss))

        torch.save(net.state_dict(), "State_Dictionaries/chest_artifact_remover")

        net.eval()
        test_loss = 0
        with torch.no_grad():
            for b in range(num_batches):

                X = X_test[b::num_batches,:,:,:].to(dev)
                Y = Y_test[b::num_batches,:,:,:].to(dev)

                out = net(X)

                loss = torch.mean((out - Y)**2)
                test_loss += loss.item()/num_batches

        test_losses.append(test_loss)
        print(" Testing Loss: {}".format(test_loss))
        
        plt.semilogy(np.array(train_losses), label = "training")
        plt.semilogy(np.array(test_losses), label = "testing")
        plt.legend()
        plt.savefig("Figures/chest_artifact_removal_losses.png")
        plt.close()








