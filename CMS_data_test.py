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

path = r"C:\Users\kaczb\Desktop\DoubleMuon_Run2022C_skim.root"
DY = uproot.open(path)

DY.keys()

#%%

import uproot

# Assuming 'file.root' contains your histogram
file = uproot.open(r"C:\Users\kaczb\Desktop\DoubleMuon_Run2022C_skim.root")

# Assuming 'histogram_key' is the key of your histogram in the file
histogram_key = "invM_ll"

# Access the histogram using the key
histogram = file[histogram_key]

# Convert the histogram to a hist object
hist_object = histogram.to_hist()

#%%

hist_object[2]

#%%

import matplotlib.pyplot as plt

# Assuming hist_object represents your histogram object

# Plotting the histogram using matplotlib
plt.figure(figsize=(10, 6))
plt.hist(hist_object, bins=100, range=(0, 200), edgecolor='black')  # Adjust bins and range as needed
plt.xlabel('X-axis Label')
plt.ylabel('Y-axis Label')
plt.title('Histogram Plot')
plt.grid(True)
plt.show()



#%%

tree_mass = DY['invM_ll;1']
tree_MET = DY['MET_pt;1']
tree_pt0, tree_pt1 = DY['lepton0pt'], DY['lepton1pt']
tree_bjets = DY['n_bJets']
tree = DY['Runs']

#%%

tree['run'].arrays()

#%%

nein = tree.to_hist

#%%

len(nein)

#%%

non = tree_mass.to_numpy(flow=False, dd=True)
#%%

plt.hist(non[0], bins = [  0.,   2.,   4.,   6.,   8.,  10.,  12.,  14.,  16.,  18.,  20.,
         22.,  24.,  26.,  28.,  30.,  32.,  34.,  36.,  38.,  40.,  42.,
         44.,  46.,  48.,  50.,  52.,  54.,  56.,  58.,  60.,  62.,  64.,
         66.,  68.,  70.,  72.,  74.,  76.,  78.,  80.,  82.,  84.,  86.,
         88.,  90.,  92.,  94.,  96.,  98., 100., 102., 104., 106., 108.,
        110., 112., 114., 116., 118., 120., 122., 124., 126., 128., 130.,
        132., 134., 136., 138., 140., 142., 144., 146., 148., 150., 152.,
        154., 156., 158., 160., 162., 164., 166., 168., 170., 172., 174.,
        176., 178., 180., 182., 184., 186., 188., 190., 192., 194., 196.,
        198., 200.])

#%%

th1_object = tree_mass.__dict__['_bases'][0]

# Access and print the attributes of the TH1 object using __dict__
th1_attributes = th1_object.__dict__
print(th1_attributes)
#%%
test = th1_object.to_pyroot

#%%

len(test)

#%%

DY = uproot.open(path)
tree1 = DY['LPHY2131analysis/WeakBosonsAnalysis']
branches = tree1.arrays()