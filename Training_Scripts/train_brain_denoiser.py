import numpy as np
import torch
import matplotlib.pyplot as plt

import sys
sys.path.append("./")
from networks import *

if __name__ == "__main__":

    dev = torch.device('cuda:0')

    x_train = np.load("Datasets/brain_training.npy").astype(np.float32)
    train_norm = np.mean(x_train**2)**(1/2)
    x_test =  np.load("Datasets/brain_testing.npy").astype(np.float32)
    test_norm = np.mean(x_test**2)**(1/2)
    
    X_train = torch.unsqueeze(torch.from_numpy(x_train),1)
    X_test = torch.unsqueeze(torch.from_numpy(x_test),1)

    net = Denoiser().to(dev)

    num_batches = 20
    num_epochs = 3*10**2

    optimizer = torch.optim.Adam(net.parameters(), lr = 1e-4)


    train_losses = []
    test_losses = []

    inds = np.arange(x_train.shape[0])

    std = np.mean(x_train**2)**(1/2)
    


    #noise_levels = [0.001, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1]
    noise_levels = [0.2]



    for epoch in range(num_epochs):
        np.random.shuffle(inds)
        
        net.train()
        train_loss = 0

        for b in range(num_batches):

            noise_level = np.random.choice(noise_levels)

            optimizer.zero_grad()

            flipx = np.random.randint(2)
            flipy = np.random.randint(2)
            nrot = np.random.randint(4)
            X = X_train[inds[b::num_batches],:,:,:].to(dev)
            """if flipx > 0:
                X = torch.flip(X,[-2])
            if flipy > 0:
                X = torch.flip(X,[-1])
            if nrot > 0:
                X = torch.rot90(X, k=nrot, dims=(-2, -1))"""
            Xn = X + noise_level*std*torch.randn(size = X.size(), device = dev)

            out = net(Xn)

            loss = torch.mean((out - X)**2)
            train_loss += loss.item()/num_batches
            loss.backward()
            optimizer.step()
        train_losses.append(train_loss)
        print("Epoch {}".format(epoch))
        print(" Training Loss: {}".format(train_loss))
        train_rrmse = train_loss**(1/2)/train_norm
        print(" Training RRMSE: {}".format(train_rrmse))

        torch.save(net.state_dict(), "State_Dictionaries/brain_denoiser")

        net.eval()
        test_loss = 0
        with torch.no_grad():
            for b in range(num_batches):

                noise_level = np.random.choice(noise_levels)
                X = X_test[b::num_batches,:,:,:].to(dev)
                Xn = X + noise_level*std*torch.randn(size = X.size(), device = dev)

                out = net(Xn)

                loss = torch.mean((out - X)**2)
                test_loss += loss.item()/num_batches

        test_losses.append(test_loss)
        print(" Testing Loss: {}".format(test_loss))
        test_rrmse = test_loss**(1/2)/test_norm
        print(" Testing RRMSE: {}".format(test_rrmse))
        
        plt.semilogy(np.array(train_losses), label = "training")
        plt.semilogy(np.array(test_losses), label = "testing")
        plt.legend()
        plt.savefig("Figures/brain_denoiser_losses.png")
        plt.close()








