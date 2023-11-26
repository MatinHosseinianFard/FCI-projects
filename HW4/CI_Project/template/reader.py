# reader.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Justin Lizama (jlizama2@illinois.edu) on 09/28/2018
# Modified by Mahir Morshed for the spring 2021 semester
# Modified by Joao Marques (jmc12@illinois.edu)for the fall 2021 semester

"""
This file is responsible for providing functions for reading the files
"""
from os import listdir
import numpy as np
import pickle
from torch.utils.data import Dataset, DataLoader
import torch
import random

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def load_dataset(filename,full = False):
    A = unpickle(filename)#np.loadtxt('data_batch_1')
    X = A[b'data']
    Y = A[b'labels'].astype(int)
    if(not full):
        test_size = int(0.25 * len(X)) # set aside 25% for testing
        X_test = X[:test_size]
        Y_test = Y[:test_size]
        X = X[test_size:]
        Y = Y[test_size:]

        return X,Y,X_test,Y_test
    else:
        return X,Y

class MP5Dataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, X,y):
        """
        Args:
            X [np.array]: features vector
            y [np.array]: labels vector          
        """
        self.data = X
        self.labels = y

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        features = self.data[idx,:]
        label = self.labels[idx]
        sample = {'features': features,'labels': label}

        return sample

def init_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
