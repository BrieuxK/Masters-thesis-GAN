# -*- coding: utf-8 -*-
"""
Created on Fri May 17 17:36:16 2024

@author: kaczb
"""

from __future__ import print_function, division
from keras.datasets import mnist

from keras.layers import Input, Dense, Reshape, Flatten, BatchNormalization, LeakyReLU, Embedding, Dropout, multiply
from keras.models import Sequential, Model
from keras.optimizers import Adam, SGD
from keras.callbacks import LearningRateScheduler, ReduceLROnPlateau
from keras import regularizers
from keras import backend as K
from tensorflow.keras.initializers import HeNormal
from keras.callbacks import Callback
from sklearn.preprocessing import MinMaxScaler

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import awkward as ak
from tqdm import tqdm
import random
#import pandas as pd

import uproot
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

#%%

path = r'/home/brieux/Bureau/DoubleMuon_Run2022C_skim.root'
DY = uproot.open(path)

DY.keys()

#%%

import uproot
import matplotlib.pyplot as plt

# Assuming 'file.root' contains your histogram
file = uproot.open(path)

# Assuming 'histogram_key' is the key of your histogram in the file
histogram_key = ["invM_ll", "MET_pt", "lepton0pt", "lepton1pt", "n_bJets"]

# print(type(file['invM_ll']))
cas = 1
for key in histogram_key:
    histogram = file[key]
    
    # Convert the histogram to a hist object
    hist_object = histogram.to_numpy()
    
    bin_counts = hist_object[0]
    bin_edges = hist_object[1]
    
    plt.figure(figsize=(10, 6))
    plt.hist(bin_edges[:-1], bins=bin_edges, weights=bin_counts, color='b')
    plt.xlabel('Gev/c²', fontsize = 15)
    plt.ylabel('Probability', fontsize = 15)
    
    if cas != 5:
        plt.xlim(0,130)
    elif cas == 5:
        plt.xlim(0, 6)
    plt.grid(True)
    # plt.xlim(0,130)
    plt.show()


#%%
#invM_ll, MET_pt, lepton0pt, lepton1pt, n_bJets

tree_mass = DY['invM_ll;1']
tree_MET = DY['MET_pt;1']
tree_pt0, tree_pt1 = DY['lepton0pt'], DY['lepton1pt']
tree_bjets = DY['n_bJets']
tree = DY['Runs']
