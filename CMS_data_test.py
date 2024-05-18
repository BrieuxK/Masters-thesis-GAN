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

# Assuming 'file.root' contains your histogram
file = uproot.open(path)

# Assuming 'histogram_key' is the key of your histogram in the file
histogram_key = "invM_ll"

# print(type(file['invM_ll']))

# Access the histogram using the key
histogram = file[histogram_key]

# Convert the histogram to a hist object
hist_object = histogram.to_numpy()

#%%

hist_object[0][0]

#%%

import matplotlib.pyplot as plt

# Assuming hist_tuple is your histogram tuple from C++
# hist_tuple[0] contains the bin counts (number of events per bin)
# hist_tuple[1] contains the bin edges

# Extract bin counts and edges
bin_counts = hist_object[0]
bin_edges = hist_object[1]

# Print bin counts and edges for verification
print("Bin counts:", bin_counts)
print("Bin edges:", bin_edges)

# Plot the histogram using matplotlib
plt.figure(figsize=(10, 6))
plt.hist(bin_edges[:-1], bins=bin_edges, weights=bin_counts, edgecolor='black', histtype='step')
plt.xlabel('Bin')
plt.ylabel('Counts')
plt.title('Histogram Plot')
plt.grid(True)
plt.show()


#%%

tree_mass = DY['invM_ll;1']
tree_MET = DY['MET_pt;1']
tree_pt0, tree_pt1 = DY['lepton0pt'], DY['lepton1pt']
tree_bjets = DY['n_bJets']
tree = DY['Runs']
