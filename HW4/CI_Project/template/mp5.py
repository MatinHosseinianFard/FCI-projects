# mp3.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Justin Lizama (jlizama2@illinois.edu) on 10/27/2018
# Modified by Mahir Morshed for the spring 2021 semester

import sys
import argparse
import configparser
import copy

import numpy as np
import torch
import pickle
import reader
import neuralnet_part1 as p1
import neuralnet_part2 as p2
import neuralnet_leaderboard as p3
from utils import compute_accuracies, get_parameter_counts

"""
This file contains the main application that is run for this MP.
"""

def main(args):
    reader.init_seeds(args.seed)

    train_set, train_labels, dev_set, dev_labels = reader.load_dataset(args.dataset_file)
    if(args.part == 1):
        p = p1
    elif(args.part == 2):
        p = p2
    elif(args.part == 3):
        p = p3

    # print("train_set : ", train_set.shape)
    train_set = torch.tensor(train_set, dtype=torch.float32)
    train_labels = torch.tensor(train_labels, dtype=torch.int64)
    dev_set = torch.tensor(dev_set, dtype=torch.float32)
    L, predicted_labels, net = p.fit(train_set, train_labels, dev_set, args.epochs)
    # print(L)
    # print(predicted_labels)
    # print(dev_labels)
    # count = 0
    # for i in range(len(predicted_labels)):
    #     if predicted_labels[i] == dev_labels[i]:
    #         count += 1
    # print(count, "/", len(predicted_labels))
    if(args.part == 3):
        torch.save(net, "net.model")
        torch.save(net.state_dict(),'state_dict.state')
    # pickle.dump(net,open('net.model','wb'))
    l = net.step(train_set[-1,:].unsqueeze(0), train_labels[-1].unsqueeze(0))
    assert type(l) == float, "your step function returned the loss as {} instead of a scalar of type float. Make sure to use .detach().cpu().numpy() on the loss before you return it in step function and to convert the output to a python float!".format(type(l))
    assert type(predicted_labels) == np.ndarray, "your fit function returned the predicted labels as {} instead of np.ndarray. Make sure to use .detach().cpu().numpy() on the network output - and don't forget to argmax it!".format(type(predicted_labels))
    assert type(L) == list,"your fit function returned the losses as {} instead of list. Make sure you are returning a list of losses (with length equal to the number of epochs)".format(type(L))
    accuracy, conf_m = compute_accuracies(predicted_labels, dev_set, dev_labels)

    print("\n Accuracy:", round(accuracy, 3))
    print("\nConfusion Matrix = \n {}".format(conf_m))
    num_parameters,params = get_parameter_counts(net)
    print('\nparameters = {} \n'.format(num_parameters))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='CS440/ECE448 MP5: Neural Nets and PyTorch')

    parser.add_argument('--dataset', dest='dataset_file', type=str, default = './data/mp5_data',
                        help='directory containing the training data')
    parser.add_argument('--epochs',dest="epochs", type=int, default = 50,
                        help='Training Epochs: default 50')
    parser.add_argument('--part', dest="part", type=int, default=1,
                        help='1 for Part 1, 2 for  Part 2 and 3 for leaderboard')
    parser.add_argument('--seed', dest="seed", type=int, default=42,
                        help='seed source for randomness')


    args = parser.parse_args()
    main(args)
