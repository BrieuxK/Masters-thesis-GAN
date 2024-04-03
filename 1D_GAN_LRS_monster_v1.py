# -*- coding: utf-8 -*-
"""
Created on Sun Feb 25 13:43:32 2024

@author: kaczb
"""
from keras.layers import Input, Dense, Reshape, Flatten, BatchNormalization, LeakyReLU
from keras.models import Sequential, Model
from keras.optimizers import Adam, SGD
from keras.callbacks import LearningRateScheduler, ReduceLROnPlateau
from keras import regularizers
from keras import backend as K
from tensorflow.keras.initializers import HeNormal
from keras.callbacks import Callback

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import awkward as ak

import uproot
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

path = r'C:\Users\kaczb\Desktop\Analysis_DY.root'
DY = uproot.open(path)
tree1 = DY['LPHY2131analysis/WeakBosonsAnalysis']
branches = tree1.arrays()

#%%
        
def clr(half_batch, step_size, base_lr, max_lr, mode, gamma = 0.999):
    cycle = np.floor(1 + half_batch / (2 * step_size))
    x = np.abs(half_batch / step_size - 2 * cycle + 1)
    if mode == 'triangular':
        out = base_lr + (max_lr - base_lr) * np.maximum(0, (1 - x))
    if mode == 'triangular2':
        out = base_lr + (max_lr - base_lr) * np.maximum(0, (1 - x))/float(2**(cycle-1))
    if mode == 'exp_range':
        out = base_lr + (max_lr - base_lr) * np.maximum(0, (1 - x))*gamma**(half_batch)
    return out

def ReduceLROP(best_loss, current, learning_rate, factor, patience, min_lr, model, epoch):
    global wait

    if current < best_loss :
        best_loss = current
        wait = 0
    else:
        wait += 1
        if wait >= patience :
            new_lr = max(learning_rate * factor, min_lr)
            wait = 0
            return new_lr, best_loss
    return learning_rate, best_loss
        

def selection(events, branches=None):
    # first filter on the number of objects
    filter = (events.nMuons == 2) & (events.nElectrons == 0)
    selected = events[filter]
    """
    # then apply a cut on the muon Pt
    filter = (selected.MuonsPt[:,0]>10) & (selected.MuonsPt[:,1]>2) 
    selected = selected[filter]
    # cut on the isolation
    filter = (selected.MuonIsolation[:,0]<0.5) & (selected.MuonIsolation[:,1]<0.5) 
    selected = selected[filter]
    """
    if branches:
        return selected[branches]
    else:
        return selected

def analyzeTree(tree, branch, simpleselection=None, selection=None, index=None, step_size="10 MB"):
    selected = ak.Array([])
    for batch in tree.iterate(step_size=step_size):
        if simpleselection:
            batchsel = batch[simpleselection(batch)][branch]
        elif selection:
            batchsel = selection(batch,branch)
        else:
            batchsel = batch[branch]
        if index is not None: # if there is an index, we select that column
            batchsel = batchsel[:,index]
        elif (type(branch) is str): # otherwise, we flatten the selection
            batchsel = ak.flatten(batchsel, axis=None)
        else: # unless branch is a list, in which case we do nothing.
            pass
        selected = ak.concatenate([selected,batchsel])
    return selected

def rescale(x, i_min, i_max, f_min, f_max):
    rescaled_value = ((x - i_min) / (i_max - i_min)) * (f_max - f_min) + f_min
    return rescaled_value
"""
def performance(minmax, values, batch_size, epoch):
    if batch_size == 2:
        plt.hist(values, bins=25, density=False, histtype='step', color='black', alpha=1, linewidth = 2)
        #plt.hist(np.concatenate([selected_simu["MuonsPt"][:,0], selected_simu["MuonsPt"][:,0], selected_simu["MuonsPt"][:,0]]), bins = 25, color = 'b')
        plt.title("Generated vs Input (Epochs: {})".format(epoch))
        plt.show()
    else:
        pass
"""
#selected_simu = analyzeTree(tree1,['MuonsPt', 'MuonsEta', 'MuonsPhi', 'invMass', 'nJets'],selection=selection)
selected_simu = analyzeTree(tree1,['MuonsPt'],selection=selection)

sc = StandardScaler()


#%%

keys_f = ['MuonsPt']

selected = selection(branches)
brl = selected.to_list()

ld = []
for i in range(len(brl)):
    ld.append(brl[i]["MuonsPt"][0])
    #ld.append(brl[i]["MuonsPt"][1])
    #ld.append(brl[i]["MuonsEta"])
my_array = np.array(ld).reshape(3413, len(keys_f))


#%%

img_rows = 1 
img_cols = 1  
img_shape = (img_rows, img_cols)

def build_generator():

    noise_shape = (10,) #1D array of size 100 (latent vector / noise)
    node = 64

    model = Sequential()

    model.add(Dense(node, input_shape=noise_shape))#, kernel_initializer=HeNormal()))#, kernel_regularizer=regularizers.l2(0.01))) #initial : 256
    model.add(LeakyReLU(alpha=0.4))
    model.add(BatchNormalization(momentum=0.9))

    model.add(Dense(node))#, kernel_initializer=HeNormal()))
    model.add(LeakyReLU(alpha=0.4))
    model.add(BatchNormalization(momentum=0.9))

    model.add(Dense(node))#, kernel_initializer=HeNormal()))
    model.add(LeakyReLU(alpha=0.4))
    model.add(BatchNormalization(momentum=0.9))
    
    #model.add(Dense(node))
    #model.add(LeakyReLU(alpha=0.4))
    #model.add(BatchNormalization(momentum=0.5))
             
    model.add(Dense(node))#, kernel_initializer=HeNormal()))
    model.add(LeakyReLU(alpha=0.4))
    model.add(BatchNormalization(momentum=0.5))
    
    model.add(Dense(np.prod(img_shape), activation='sigmoid')) #according to finance paper, here : "linear"
    model.add(Reshape(img_shape))
    
    model.summary()

    noise = Input(shape=noise_shape)
    img = model(noise)    #Generated image

    return Model(noise, img)

def build_discriminator():
    
    node = 64
    model = Sequential()

    model.add(Flatten(input_shape=img_shape))
    model.add(Dense(node))#, input_shape=img_shape, kernel_initializer=HeNormal()))
    model.add(LeakyReLU(alpha=0.4))

    model.add(Dense(node))#, kernel_initializer=HeNormal()))
    model.add(LeakyReLU(alpha=0.4))

    #model.add(Dense(node))
    #model.add(LeakyReLU(alpha=0.4))

    model.add(Dense(node))#, kernel_initializer=HeNormal()))
    model.add(LeakyReLU(alpha=0.4))

    model.add(Dense(node))#, kernel_initializer=HeNormal()))
    model.add(LeakyReLU(alpha=0.4))

    model.add(Dense(1, activation='sigmoid'))  # Using 'sigmoid' for binary classification
    model.summary()

    img = Input(shape=img_shape)
    validity = model(img)

    return Model(img, validity)

def train(data_amount, epochs, batch_size, save_interval=2000, callbacks=None):
    global best_loss
    
    learning_rate = 0.9
    optimizer = SGD(learning_rate, momentum=0.8, nesterov=True, clipvalue=1.0)

    discriminator = build_discriminator()
    discriminator.compile(loss='binary_crossentropy',
                          optimizer=optimizer,
                          metrics=['accuracy'])

    generator = build_generator()
    generator.compile(loss='binary_crossentropy', optimizer=optimizer)

    z = Input(shape=(10,))
    img = generator(z)

    discriminator.trainable = False
    valid = discriminator(img)

    combined = Model(z, valid)
    combined.compile(loss='binary_crossentropy', optimizer=optimizer)
    
    values = [] #tool for rescaling output
    ce_loss = [] #cross-entropy loss for G
    ce_loss1 = [] #cross-entropy loss for D
    lr_list = [] #list with learning rate values
    
    x_400 = [] #nbr of epoch*400 for LR plot

    y = np.ones(len(my_array))
    X_train_raw, x_test, y_train, y_test = train_test_split(my_array, y)
    
    sc = StandardScaler()
    sc.fit(X_train_raw)
    X_train_raw = sc.transform(X_train_raw)
    x_test = sc.transform(x_test)

    X_train = []
    for i in range(len(X_train_raw)):
        X_train.append(rescale(X_train_raw[i][0], min(X_train_raw)[0], max(X_train_raw)[0], 0, 1))

    half_batch = int(batch_size / 2)
    
    
    for epoch in range(epochs):
        
        # if epoch < 2000:
        #     pass
        # else:
        #     learning_rate = 1.1
        
        # if epoch < 5000:
        #     learning_rate = 0.1
        # if epoch >= 5000 and learning_rate >= 1e-6:
        #     learning_rate *= np.exp(-0.0001)
        # if epoch >= 5000 and learning_rate <= 1e-6:
        #     pass
        
        #Cyclic LR
        
        learning_rate = clr(epoch, step_size= 2000, base_lr = 0.0005, max_lr=0.05, mode='triangular2') #2000, 0.001, 0.06, t2
        #2000 0.006 0.06 t2 : correct
        #5000 0.006 0.06 t2 : same
        #500 0.006 0.06 t2 : same (un tout petit peu + haut mais négligeable)
        #1000 0.006 0.06 exp_range : same
        #1000 0.008 0.02 t2 : same
        #512 nodes for all the above runs
        #2000 0.0006 0.6 t2 : pas ouf
        
        
        #LR scheduler
        
        # if epoch < 5000:
        #     pass
        # if epoch >= 1000 and learning_rate >= 1e-10:
        #     learning_rate *= np.exp(-0.001)
        # if epoch >= 1000 and learning_rate <= 1e-10:
        #     pass
        
        idx = np.random.randint(0, len(X_train), half_batch)
        imgs = np.array(X_train)[idx]

        noise = np.random.normal(0, 1, (half_batch, 10))
        gen_imgs = generator.predict(noise)

        d_loss_real = discriminator.train_on_batch(imgs, np.ones((half_batch, 1)))
        d_loss_fake = discriminator.train_on_batch(gen_imgs, np.zeros((half_batch, 1)))
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
        
        # Reduce LR on plateau
        
        # learning_rate, best_loss = ReduceLROP(best_loss=best_loss, current=d_loss[0], learning_rate=learning_rate, factor=0.99, patience=50, min_lr=1e-10, model=combined, epoch = epoch)
        
        K.set_value(optimizer.lr, learning_rate)
        
        noise = np.random.normal(0, 1, (batch_size, 10)) 
        valid_y = np.array([1] * batch_size)
        g_loss = combined.train_on_batch(noise, valid_y)
        
        if (epochs - epoch) <= data_amount:
            stock.append(gen_imgs)
            values.append(float(gen_imgs[-1]))
        
        if epoch % save_interval == 0:
            if epoch != 0:
                plt.hist(values, bins=50, density=False, histtype='step', color='black', alpha=1, linewidth=2)
                plt.title("Generated vs Input (Epochs: {})".format(epoch))
                plt.show()

                fig, axs = plt.subplots(1, 2, figsize=(12, 6))
                axs[0].plot(ce_loss)
                axs[0].set_title('Histogram 1')
                axs[1].plot(ce_loss1)
                axs[1].set_title('Histogram 2')
                plt.tight_layout()
                plt.show()
                
                
                plt.plot(x_400, lr_list)#, [i for i in range(0, 30000, 400)])
                plt.title('Learning Rate Over Time (Epochs: {})'.format(epoch))
                plt.xlabel('Epochs')
                plt.ylabel('Learning Rate')
                plt.yscale('log')
                plt.show()

            
        if epoch % (save_interval/5) == 0:
            x_400.append(epoch)
            
            ce_loss.append(d_loss[0])
            ce_loss1.append(g_loss)
            lr_list.append(learning_rate)


#%%

#256 nodes : slightly worse
#16 nodes : no improvement vs 32
#         : slightly better, peak centered
#16 nodes with triangular1 : slightly worse
#16 nodes with exp_range with gamma = 0.999 : really bad
#16 nodes with triangular2 base_lr = 0.0001, max_lr=0.05 : best so far
#                                                  =0.08 : same result as initial
#16 nodes step size = 4000 0.0001, max_lr=0.05, mode='triangular2' : not centered but relatively high : ok-ish
#                   = 400                                          : not centered but this time toward left : meh
#                  2000, base_lr = 0.0005, max_lr=0.05, mode='triangular2' : best so far v2
#batch size = 1000                                                 : same as original result
#64 nodes 2000, base_lr = 0.0005, max_lr=0.05, mode='triangular2'  : 

#model = tf.keras.models.load_model('my_model.h5')

#%%
import time

# Record the start time
start_time = time.time()

best_loss = 1e10
wait = 0

lr_list = []
ce_loss = []
ce_loss1 = []
data_amount = 20000
stock = []
epochs = 20001
batch_size= 800
history = train(data_amount, epochs, batch_size)#, save_interval=2000)#, data_amount = 2000)
# output : nbr epochs x batch/2 x 45

end_time = time.time()

# Calculate the running time
running_time = end_time - start_time

#%%

if stock[0].shape[0] == 1:
    r_stock = [None]*len(stock)
    for i in range(len(stock)):
        stock[i] = stock[i].reshape(1,)
else:
    r_stock1 = [None]*len(stock)
    for i in range(len(stock)):
        stock[i] = stock[i][-1].reshape(1,1)  #We just take the last element for plotting
        
#%%
        
# Calculate the number of rows and columns for the subplots
num_plots = len(keys_f)


# Create a grid of subplots
fig, axes = plt.subplots(4, 4, figsize=(15, 15))

# Flatten the axes array to simplify indexing
axes = axes.flatten()

if stock[0].shape[0] == 1:
    # Loop over each element in keys_f and plot on the corresponding subplot
    for i, key in enumerate(keys_f):
        n, bins, patches = axes[i].hist(np.array(stock)[:,i], 50, density=False, facecolor='b', alpha=0.75)
        axes[i].set_xlabel(key)
        axes[i].set_ylabel('Probability')
        axes[i].grid(True)
    
    # Hide any unused subplots
    for j in range(num_plots, 4*4):
        fig.delaxes(axes[j])
    
    # Adjust layout to prevent overlap
    plt.tight_layout()
    
    # Show the plot
    plt.show()

else :
    # Loop over each element in keys_f and plot on the corresponding subplot
    for i, key in enumerate(keys_f):
        n, bins, patches = axes[i].hist(np.array(stock)[:,i], 50, density=False, facecolor='b', alpha=0.75)
        axes[i].set_xlabel(key)
        axes[i].set_ylabel('Probability')
        axes[i].grid(True)
    
    # Hide any unused subplots
    for j in range(num_plots, 4*4):
        fig.delaxes(axes[j])
    
    # Adjust layout to prevent overlap
    plt.tight_layout()
    
    # Show the plot
    plt.show()
    
print(len(np.array(stock)[:,0]))

if stock[0].shape[0] == 1:
    min_bound = min(stock)[0]
    max_bound = max(stock)[0]
else:
    min_bound = min(stock)[0]
    max_bound = max(stock)[0]
    
print("Running time:", running_time, "seconds") 

#%%

""" When activation function is not sigmoid"""

# minmax_used =[["MuonsPt1", (0, 291.8657531738281)]]

# stock_safe = stock
# resc_stock = stock
# resc_stock_safe = resc_stock
# for i in range(len(stock)): #0 to 499
#     for j in range(len(stock[i])): #0 to 11
#         resc_stock_safe[i][j] = rescale(stock_safe[i][j], min_bound, max_bound, minmax_used[j][-1][0], minmax_used[j][-1][-1])

# # Convert arrays to NumPy arrays
# resc_stock_np = np.array(resc_stock)[:, 0]
# selected_simu_np = np.array(selected_simu[key])

# # Flatten selected_simu_np if it's 2D
# if selected_simu_np.ndim == 2:
#     selected_simu_np = selected_simu_np.flatten()
    
# x = resc_stock_np #np.concatenate([resc_stock_np, resc_stock_np, resc_stock_np, resc_stock_np])
# y = np.concatenate([selected_simu_np, selected_simu_np, selected_simu_np])#, selected_simu_np, selected_simu_np])

# n_x, bins_x = np.histogram(x, bins = int(np.ceil(np.sqrt(x.size))))
# n_y, bins_y = np.histogram(y, bins = int(np.ceil(np.sqrt(y.size))))

# n_y=n_y/len(y)/len(bins_x)*len(x)*len(bins_y)

# #plotting
# plt.plot(bins_x[:-1],n_x)
# plt.plot(bins_y[:-1],n_y)
# plt.show()

#%%
    
minmax_used =[["MuonsPt1", (0, 291.8657531738281)]]

stock_safe = stock
resc_stock = stock
resc_stock_safe = resc_stock
for i in range(len(stock)): #0 to 499
    for j in range(len(stock[i])): #0 to 11
        resc_stock_safe[i][j] = rescale(stock_safe[i][j], 0, 1, minmax_used[j][-1][0], minmax_used[j][-1][-1])

# Convert arrays to NumPy arrays
resc_stock_np = np.array(resc_stock)[:, 0]
selected_simu_np = np.array(selected_simu[key])

# Flatten selected_simu_np if it's 2D
if selected_simu_np.ndim == 2:
    selected_simu_np = selected_simu_np.flatten()
    
#%%

x = resc_stock_np #np.concatenate([resc_stock_np, resc_stock_np, resc_stock_np, resc_stock_np])
y = np.concatenate([selected_simu_np])#, selected_simu_np, selected_simu_np, selected_simu_np, selected_simu_np])

n_x, bins_x = np.histogram(x, bins = int(np.ceil(np.sqrt(x.size))))
n_y, bins_y = np.histogram(y, bins = int(np.ceil(np.sqrt(y.size))))

n_y=n_y/len(y)/len(bins_x)*len(x)*len(bins_y)

#plotting
plt.plot(bins_x[:-1],n_x, color = 'black', linewidth = 2, label = 'Generated')
plt.plot(bins_y[:-1],n_y, color = 'b', linewidth = 2, label = 'Input')
plt.legend()
plt.show()

#%%
    
# Histogram Calculation
n_x, bins_x = np.histogram(x, bins=int(np.ceil(np.sqrt(x.size))))
n_y, bins_y = np.histogram(y, bins=int(np.ceil(np.sqrt(y.size))))

n_y = n_y / len(y) / len(bins_x) * len(x) * len(bins_y)

# Plotting Histograms
plt.hist(x, bins=bins_x, density=False, histtype='step', color='black', alpha=1, linewidth = 2, label='Generated')
plt.hist(y, bins=bins_y, density=False, facecolor='b', alpha=0.8, label='Input')

# Adding Labels and Legend
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.legend()

# Displaying the Plot
plt.show()

    
#%%
if batch_size == 2 :
    arg = resc_stock_np #np.concatenate([resc_stock_np, resc_stock_np, resc_stock_np, resc_stock_np])
    arg2 = np.concatenate([selected_simu_np, selected_simu_np, selected_simu_np])#, selected_simu_np, selected_simu_np])
    
    bins = np.histogram_bin_edges(np.concatenate([arg, arg2]), bins=50)
    
    # Plot the histograms using the same bins
    plt.hist(arg, bins=bins, density=False, histtype='step', color='black', alpha=1, linewidth = 2, label='Generated')
    plt.hist(arg2, bins=bins, density=False, facecolor='b', alpha=0.8, label='Input')
    
    plt.xlabel("MuonsPt")
    plt.ylabel(r'$Number\ of\ events \times 2$', fontsize=12)
    plt.grid(True)
    plt.title("Generated vs Input (Epochs: {})".format(epochs))
    plt.legend()
    plt.show()
    
    print(len(arg), len(arg2))

else:
    arg = np.squeeze(resc_stock_np) #np.concatenate([resc_stock_np, resc_stock_np, resc_stock_np, resc_stock_np])
    arg2 = np.concatenate([selected_simu_np, selected_simu_np, selected_simu_np])#, selected_simu_np, selected_simu_np])
    
    bins = np.histogram_bin_edges(np.concatenate([arg, arg2]), bins=50)
    
    # Plot the histograms using the same bins
    plt.hist(arg, bins=bins, density=False, histtype='step', color='black', alpha=1, linewidth = 2, label='Generated')
    plt.hist(arg2, bins=bins, density=False, facecolor='b', alpha=0.8, label='Input')
    
    plt.xlabel("MuonsPt")
    plt.ylabel(r'$Number\ of\ events \times 2$', fontsize=12)
    plt.grid(True)
    plt.title("Generated vs Input (Epochs: {})".format(epochs))
    plt.legend()
    plt.show()
    
    print(len(arg), len(arg2))

#np.concatenate([resc_stock_np, resc_stock_np]) #3000 epochs
#np.concatenate([resc_stock_np, resc_stock_np, resc_stock_np]) #2000 epochs
#np.concatenate([resc_stock_np, resc_stock_np, resc_stock_np, resc_stock_np]) #1500 epochs
#np.concatenate([resc_stock_np, resc_stock_np, resc_stock_np, resc_stock_np, resc_stock_np, resc_stock_np, resc_stock_np]) #1000 epochs
#np.concatenate([resc_stock_np, resc_stock_np, resc_stock_np, resc_stock_np, resc_stock_np, resc_stock_np, resc_stock_np, resc_stock_np, resc_stock_np, resc_stock_np, resc_stock_np, resc_stock_np, resc_stock_np, resc_stock_np]) #500 epochs
#%%
print(np.squeeze(resc_stock_np).shape)
print(np.concatenate([selected_simu_np, selected_simu_np, selected_simu_np]).shape)

#np.concatenate([resc_stock_np, resc_stock_np]) #3000 epochs
#np.concatenate([resc_stock_np, resc_stock_np, resc_stock_np]) #2000 epochs
#np.concatenate([resc_stock_np, resc_stock_np, resc_stock_np, resc_stock_np]) #1500 epochs
#np.concatenate([resc_stock_np, resc_stock_np, resc_stock_np, resc_stock_np, resc_stock_np, resc_stock_np, resc_stock_np]) #1000 epochs
#np.concatenate([resc_stock_np, resc_stock_np, resc_stock_np, resc_stock_np, resc_stock_np, resc_stock_np, resc_stock_np, resc_stock_np, resc_stock_np, resc_stock_np, resc_stock_np, resc_stock_np, resc_stock_np, resc_stock_np]) #500 epochs
#%%
arg = resc_stock_np #np.concatenate([resc_stock_np, resc_stock_np, resc_stock_np, resc_stock_np])
arg2 = np.concatenate([selected_simu_np, selected_simu_np, selected_simu_np, selected_simu_np, selected_simu_np])

bins_list = []
for i in range(0,200,2):
    bins_list.append(i)

bins = 50

print(len(arg), len(arg2))

# Plot the histograms using the same bins
plt.hist(arg, bins=bins_list, density=False, histtype='step', color='black', alpha=1, linewidth = 2, label='Generated')
plt.hist(arg2, bins=bins_list, density=False, facecolor='b', alpha=0.8, label='Input')

plt.xlabel("MuonsPt")
plt.ylabel(r'$Number\ of\ events \times 2$', fontsize=12)
plt.grid(True)
plt.title("Generated vs Input (Epochs: {})".format(epochs))
plt.legend()
plt.show()

#%%
# Convert arrays to NumPy arrays
resc_stock_np = np.array(resc_stock)[:, 0]
selected_simu_np = np.array(selected_simu[key])

# Flatten resc_stock_np if it's 2D
if resc_stock_np.ndim == 2:
    resc_stock_np = resc_stock_np.flatten()

# Flatten selected_simu_np if it's 2D
if selected_simu_np.ndim == 2:
    selected_simu_np = selected_simu_np.flatten()

# Compute the bins using the desired number of bins
bins = np.histogram_bin_edges(np.concatenate([resc_stock_np, selected_simu_np]), bins=50)

# Plot the histograms using the same bins
plt.hist(resc_stock_np, bins=bins, density=False, histtype='step', color='black', alpha=1, linewidth=2, label='Generated')
plt.hist(selected_simu_np, bins=bins, density=False, facecolor='b', alpha=0.8, label='Input')
#np.concatenate([selected_simu_np,selected_simu_np,selected_simu_np])

plt.xlabel("MuonsPt")
plt.ylabel('Number of events ', fontsize=12)
plt.grid(True)
plt.title("Generated vs Input")
plt.legend()
plt.show()

print(len(resc_stock_np), len(selected_simu_np) )