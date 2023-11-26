# neuralnet.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Justin Lizama (jlizama2@illinois.edu) on 10/29/2019
# Modified by Mahir Morshed for the spring 2021 semester
# Modified by Joao Marques (jmc12) for the fall 2021 semester

"""
This is the main entry point for MP5. You should only modify code
within this file, neuralnet_learderboard and neuralnet_part2.py -- the unrevised staff files will be used for all other
files and classes when code is run, so be careful to not modify anything else.
"""

import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from utils import get_dataset_from_arrays
from torch.utils.data import DataLoader


class NeuralNet(nn.Module):
    def __init__(self, lrate, loss_fn, in_size, out_size):
        """
        Initializes the layers of your neural network.

        @param lrate: learning rate for the model
        @param loss_fn: A loss function defined as follows:
            @param yhat - an (N,out_size) Tensor
            @param y - an (N,) Tensor
            @return l(x,y) an () Tensor that is the mean loss
        @param in_size: input dimension
        @param out_size: output dimension

        For Part 1 the network should have the following architecture (in terms of hidden units):

        in_size -> h ->  out_size , where  1 <= h <= 256

        We recommend setting lrate to 0.01 for part 1.

        """
        super(NeuralNet, self).__init__()
        self.loss_fn = loss_fn
        # Define hidden layer size
        hidden_size = 32
        # Define the layers of the neural network
        self.fc1 = nn.Linear(in_size, hidden_size)
        self.out = nn.Linear(hidden_size, out_size)
        # Adam optimizer
        self.optimizer = optim.Adam(self.parameters(), lr=lrate)

    def forward(self, x):
        """Performs a forward pass through your neural net (evaluates f(x)).

        @param x: an (N, in_size) Tensor
        @return y: an (N, out_size) Tensor of output from the network
        """
        x = F.relu(self.fc1(x))
        x = self.out(x)

        return x

    def step(self, x, y):
        """
        Performs one gradient step through a batch of data x with labels y.

        @param x: an (N, in_size) Tensor
        @param y: an (N,) Tensor
        @return L: total empirical risk (mean of losses) for this batch as a float (scalar)
        """
        yhat = self.forward(x)
        loss = self.loss_fn(yhat, y)

        # Zero the gradients, perform backward pass, and update the weights
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()


def fit(train_set, train_labels, dev_set, epochs, batch_size=128):
    """ Make NeuralNet object 'net' and use net.step() to train a neural net
    and net(x) to evaluate the neural net.

    @param train_set: an (N, in_size) Tensor
    @param train_labels: an (N,) Tensor
    @param dev_set: an (M,) Tensor
    @param epochs: an int, the number of epochs of training
    @param batch_size: size of each batch to train on. (default 100)

    This method _must_ work for arbitrary M and N.

    The model's performance could be sensitive to the choice of learning rate.
    We recommend trying different values in case your first choice does not seem to work well.

    @return losses: list of floats containing the total loss at the beginning and after each epoch.
            Ensure that len(losses) == epochs.
    @return yhats: an (M,) NumPy array of binary labels for dev_set
    @return net: a NeuralNet object
    """
    # Get input size and number of unique labels
    in_size = train_set.shape[1]
    out_size = len(torch.unique(train_labels))

    # Initialize the neural network
    net = NeuralNet(lrate=0.01, loss_fn=nn.CrossEntropyLoss(),
                    in_size=in_size, out_size=out_size)

    # Standardize train data
    train_mean = train_set.mean(axis=0)
    train_std = train_set.std(axis=0)
    train_set = (train_set - train_mean) / train_std

    # Create a PyTorch dataset and dataloader
    train_dataset = get_dataset_from_arrays(train_set, train_labels)
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True)

    losses = []

    # Training loop
    for epoch in tqdm(range(epochs)):
        total_loss = 0.0
        # Iterate over batches
        for batch in train_loader:
            loss = net.step(batch['features'], batch['labels'])
            total_loss += loss

        avg_loss = total_loss / len(train_loader)
        losses.append(avg_loss)

    # Standardize dev data
    dev_mean = dev_set.mean(axis=0)
    dev_std = dev_set.std(axis=0)

    dev_set = (dev_set - dev_mean) / dev_std

    # Evaluate on dev_set
    net.eval()
    with torch.no_grad():
        dev_set = dev_set.detach().clone()
        yhats = np.argmax(net(dev_set).detach().numpy(), axis=1)

    return losses, yhats, net
