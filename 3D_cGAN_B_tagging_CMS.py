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
import pandas as pd

import uproot
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

path = r'C:\Users\kaczb\Desktop\dyb_test2.root'
DY = uproot.open(path)
tree1 = DY['LPHY2131analysis/WeakBosonsAnalysis']
branches = tree1.arrays()

# DY.keys()

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

#%%

selected_simu = analyzeTree(tree1,['MuonsPt', 'invMass', 'MET_pt', 'JetsB'],selection=selection) #'MET_pt' ?
selected_jets = analyzeTree(tree1,['nJets'],selection=selection) #'MET_pt' ?

sc = StandardScaler()

scaler = MinMaxScaler()
#%%
wui = []
for i in selected_simu['JetsB']:
    wui.append(np.array(i))

b_tag = []
for i in wui:
    if i.size == 0:
        b_tag.append(0)
    else:
        cnt = 0
        for j in i:
            if j == 1:
                cnt += 1
        b_tag.append(cnt)

b_tag = np.array(b_tag)
#%%
keys_f = ['MuonsPt', 'invMass', 'MET_pt']

selected = selection(branches)
brl = selected.to_list()

ld = []
for i in range(len(brl)):
    ld.append(brl[i]["MuonsPt"][0])
    ld.append(brl[i]["invMass"])
    ld.append(brl[i]["MET_pt"])
    # ld.append(b_tag[i])

my_array_old = np.array(ld).reshape(len(brl), len(keys_f))

my_array = my_array_old

#%%

type(my_array)

#%%

import pandas as pd

path = r"C:\Users\kaczb\Desktop\DoubleMuon_Run2022C.parquet"
df = pd.read_parquet(path, engine='pyarrow')

mask = (df.iloc[:, :4] == 0.000000).any(axis=1)
df_filtered = df[~mask]

#Values are not swapped, if the smallest is in 1st col, it is simply replaced by the largest, the smallest is removed from the df
if 'muon0pt' in df_filtered.columns and 'muon1pt' in df_filtered.columns:
    df_filtered['muon0pt'] = np.maximum(df_filtered['muon0pt'], df_filtered['muon1pt'])

df_filtered.drop(columns=['muon1pt'], inplace=True)

#%%

numpy_array = df_filtered.to_numpy()

sample = numpy_array[:, :-1]
label = numpy_array[:, -1]

label[label >= 1] = 1

#%%
sample.shape

#Reducing the shape of the sample
size = 25000

my_array = sample[:size]
label_df = label[:size]

#%%

print(type(sample), type(label_df))

#%%
#Label selection

threshold = 0

labels_01 = np.where(b_tag > threshold, 1, 0) #1 if b_tag > threshold, 0 otherwise

# for i in range(50):
#     print(i, labels_01[i])

#%%

print(type(labels_01))

#%%

#remove batch_norm => bad results
#remove 1 layer => same
#add 2 layers => invMass and MuonPt looking very similar
#128 nodes

# Input shape
img_rows = 3
img_cols = 1
img_shape = (img_rows, img_cols)
num_classes = 2
latent_dim = 25

def build_generator(latent_dim):
    
    node = 32

    model = Sequential()

    model.add(Dense(node, input_dim=latent_dim, kernel_initializer=HeNormal(), bias_initializer='zeros'))
    model.add(LeakyReLU(alpha=0.4))
    model.add(BatchNormalization(momentum=0.5))
 
    model.add(Dense(node, kernel_initializer=HeNormal(), bias_initializer='zeros'))
    model.add(LeakyReLU(alpha=0.4))
    model.add(BatchNormalization(momentum=0.5))
 
    model.add(Dense(node, kernel_initializer=HeNormal(), bias_initializer='zeros'))
    model.add(LeakyReLU(alpha=0.4))
    model.add(BatchNormalization(momentum=0.5))
    
    # model.add(Dense(node, kernel_initializer=HeNormal(), bias_initializer='zeros'))
    # model.add(LeakyReLU(alpha=0.4))
    # model.add(BatchNormalization(momentum=0.5))
    
    model.add(Dense(node, kernel_initializer=HeNormal(), bias_initializer='zeros'))
    model.add(LeakyReLU(alpha=0.4))
    model.add(BatchNormalization(momentum=0.5))
    
    model.add(Dense(np.prod(img_shape), activation='sigmoid')) 
    model.add(Reshape(img_shape))

    model.summary()

    noise = Input(shape=(latent_dim,))
    label = Input(shape=(1,), dtype='int32')
    label_embedding = Flatten()(Embedding(num_classes, latent_dim)(label))

    model_input = multiply([noise, label_embedding])
    img = model(model_input)

    return Model([noise, label], img)

def build_discriminator():

    model = Sequential()
    
    node = 32

    # model.add(Flatten(input_shape=img_shape))
    # model.add(Dense(node, input_shape=img_shape, kernel_initializer=HeNormal(), bias_initializer='zeros'))
    model.add(Dense(node, input_dim=np.prod(img_shape), kernel_initializer=HeNormal(), bias_initializer='zeros'))
    model.add(LeakyReLU(alpha=0.4))

    model.add(Dense(node, kernel_initializer=HeNormal(), bias_initializer='zeros'))
    model.add(LeakyReLU(alpha=0.4))

    model.add(Dense(node, kernel_initializer=HeNormal(), bias_initializer='zeros'))
    model.add(LeakyReLU(alpha=0.4))

    model.add(Dense(node, kernel_initializer=HeNormal(), bias_initializer='zeros'))
    model.add(LeakyReLU(alpha=0.4))
    
    # model.add(Dense(node, kernel_initializer=HeNormal(), bias_initializer='zeros'))
    # model.add(LeakyReLU(alpha=0.4))

    model.add(Dense(1, activation='sigmoid'))  # Using 'sigmoid' for binary classification
    model.summary()

    img = Input(shape=img_shape)
    label = Input(shape=(1,), dtype='int32')

    label_embedding = Flatten()(Embedding(num_classes, np.prod(img_shape))(label))
    flat_img = Flatten()(img)

    model_input = multiply([flat_img, label_embedding])

    validity = model(model_input)

    return Model([img, label], validity)

def train(epochs, batch_size=128):
    global best_loss
    global labels_01
    
    # selected_simu_np = np.array(selected_simu["MuonsPt"][:,0])
    # selected_simu_np1 = np.array(selected_simu["invMass"])
    # selected_simu_np2 = np.array(selected_simu["MET_pt"])
    
    learning_rate = 0.001
    optimizer = SGD(learning_rate, momentum=0.8, nesterov=True, clipvalue=1.0)

    # Build and compile the discriminator
    discriminator = build_discriminator()
    discriminator.compile(loss=['binary_crossentropy'],
        optimizer=optimizer,
        metrics=['accuracy'])

    # Build the generator
    generator = build_generator(latent_dim)

    # The generator takes noise and the target label as input
    # and generates the corresponding digit of that label
    noise = Input(shape=(latent_dim,))
    label = Input(shape=(1,))
    img = generator([noise, label])

    # For the combined model we will only train the generator
    discriminator.trainable = False

    # The discriminator takes generated image as input and determines validity
    # and the label of that image
    valid = discriminator([img, label])

    # The combined model  (stacked generator and discriminator)
    # Trains generator to fool discriminator
    combined = Model([noise, label], valid)
    combined.compile(loss=['binary_crossentropy'],
        optimizer=optimizer)
    
    # y = np.ones(len(my_array))
    
    y = label_df
    
    X_train_raw, x_test, y_train, y_test = train_test_split(my_array, y, test_size=0.1)
    
    X_train = scaler.fit_transform(X_train_raw)
    
    half_batch = int(batch_size / 2)

    # Adversarial ground truths
    valid = np.ones((half_batch, 1)) #batch_size ?
    # fake = np.zeros((batch_size, 1))
    
    values = [[], [], []] #tool for rescaling output
    ce_loss = [] #cross-entropy loss for G
    ce_loss1 = [] #cross-entropy loss for D
    lr_list = [] #list with learning rate values
    
    x_400 = [] #nbr of epoch*400 for LR plot
    x_vary = [] #nbr of epoch for loss functions

    for epoch in tqdm(range(epochs)):
        
        #Cyclic LR
        
        learning_rate = clr(epoch, step_size= 2000, base_lr = 0.00005, max_lr=0.05, mode='triangular') #2000, 0.001, 0.06, t2
    
        idx = np.random.randint(0, len(X_train), batch_size)
        imgs, labels = X_train[idx], y_train[idx]

        noise = np.random.normal(0, 1, (batch_size, latent_dim))
        gen_imgs = generator.predict([noise, labels], verbose=0)
        
        d_loss_real = discriminator.train_on_batch([imgs, labels], np.ones((batch_size, 1)))
        d_loss_fake = discriminator.train_on_batch([gen_imgs, labels], np.zeros((batch_size, 1)))

        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
        
        K.set_value(optimizer.lr, learning_rate)
        
        noise = np.random.normal(0, 1, (batch_size, latent_dim)) #batch_size ?
        valid_y = np.array([1] * batch_size) #batch_size ?
        
        sampled_labels = np.random.randint(0, 2, batch_size).reshape(-1, 1) #-1,1 ? #batch_size ?
        
        # idx_bis = np.random.randint(0, 2, batch_size)
        # sampled_labels = labels[idx_bis].reshape(-1, 1)
        
        
        g_loss = combined.train_on_batch([noise, sampled_labels], valid_y)
        
        stock[0].append(float(gen_imgs[0][0]))
        stock[1].append(float(gen_imgs[0][1]))
        stock[2].append(float(gen_imgs[0][2]))
        
        stockbis[0].append(float(gen_imgs[1][0]))
        stockbis[1].append(float(gen_imgs[1][1]))
        stockbis[2].append(float(gen_imgs[1][2]))
        
        # Append in the first sub-list of "values" the data generated for the first key, and same for second sub-list
        values[0].append(float(gen_imgs[-1][0]))
        values[1].append(float(gen_imgs[-1][1]))
        values[2].append(float(gen_imgs[-1][2]))
        
        
        if epoch % 2000  == 0:
            if epoch != 0:
                fig, axs = plt.subplots(1, 3, figsize=(12, 6))

                # Plot the first histogram on the first subplot
                axs[0].hist(values[0], bins=50, density=False, histtype='step', color='black', alpha=1, linewidth=2)
                axs[0].set_title("{} : Generated vs Input (Epochs: {})".format(keys_f[0], epoch))
                
                # Plot the second histogram on the second subplot
                axs[1].hist(values[1], bins=50, density=False, histtype='step', color='black', alpha=1, linewidth=2)
                axs[1].set_title("{} : Generated vs Input (Epochs: {})".format(keys_f[1], epoch))
                
                axs[2].hist(values[2], bins=50, density=False, histtype='step', color='black', alpha=1, linewidth=2)
                axs[2].set_title("{} : Generated vs Input (Epochs: {})".format(keys_f[2], epoch))
                
                plt.tight_layout()
                plt.show()

                fig, axs = plt.subplots(1, 2, figsize=(12, 6))
                axs[0].plot(ce_loss)
                axs[0].set_title('Discriminator loss')
                axs[1].plot(ce_loss1)
                axs[1].set_title('Generator loss')
                plt.tight_layout()
                plt.show()
                
                
                plt.plot(x_400, lr_list)#, [i for i in range(0, 30000, 400)])
                plt.title('Learning Rate Over Time (Epochs: {})'.format(epoch))
                plt.xlabel('Epochs')
                plt.ylabel('Learning Rate')
                plt.yscale('log')
                plt.show()
                
                #####################################################################
                
                # fig, axs = plt.subplots(1, 2, figsize=(12, 6))  # Two subplots, one for each class

                # # Flatten and concatenate generated images for class "0"
                # gen_imgs_class_0 = np.concatenate([gen_imgs[i].flatten() for i in range(len(gen_imgs)) if sampled_labels[i] == 0])
            
                # # Plot histogram for class "0"
                # axs[0].hist(gen_imgs_class_0, bins=50, density=False, histtype='step', color='black', alpha=1, linewidth=2)
                # axs[0].set_title("Class 0: Generated vs Input (Epochs: {})".format(epoch))
            
                # # Flatten and concatenate generated images for class "1"
                # gen_imgs_class_1 = np.concatenate([gen_imgs[i].flatten() for i in range(len(gen_imgs)) if sampled_labels[i] == 1])
            
                # # Plot histogram for class "1"
                # axs[1].hist(gen_imgs_class_1, bins=50, density=False, histtype='step', color='black', alpha=1, linewidth=2)
                # axs[1].set_title("Class 1: Generated vs Input (Epochs: {})".format(epoch))
            
                # plt.tight_layout()
                # plt.show()

        
        if epoch <= 2000:
            if epoch % 10 == 0 :
                x_vary.append(epoch)
                ce_loss.append(d_loss[0])
                ce_loss1.append(g_loss)
        else:
            if epoch % (400) == 0:
                x_vary.append(epoch)
                ce_loss.append(d_loss[0])
                ce_loss1.append(g_loss)
                
        if epoch % 400 == 0:
            x_400.append(epoch)
            lr_list.append(learning_rate)

#%%
import time

# Record the start time
start_time = time.time()

best_loss = 1e10
wait = 0

batch_test_1 = []
batch_test_2 = []
lr_list = []
ce_loss = []
ce_loss1 = []
stock = [[], [], []]
stockbis = [[], [], []]
epochs = 10000
batch_size= 600
train(epochs, batch_size)#, save_interval=2000)#, data_amount = 2000)
# output : nbr epochs x batch/2 x 45

end_time = time.time()

# Calculate the running time
running_time = end_time - start_time
 #%%

stock = np.transpose(np.array(stock))
stockbis = np.transpose(np.array(stockbis))

print("Running time:", running_time, "seconds") 

rescaled_test = scaler.inverse_transform(stock)
rescaled_testbis = scaler.inverse_transform(stockbis)

#%%

resc_stock_np = np.array(rescaled_test)
resc_stock_npbis = np.array(rescaled_testbis)

selected_simu_np = np.array(selected_simu["MuonsPt"][:,0])
selected_simu_np1 = np.array(selected_simu["invMass"])
selected_simu_np2 = np.array(selected_simu["MET_pt"])

#%%

resc_stock_safe = resc_stock_np

fig, axes = plt.subplots(1, 3, figsize=(12, 6))
axes = axes.flatten()

# Adjust the vertical gap between subplots
fig.subplots_adjust(hspace=0.2)

arg = np.squeeze(resc_stock_safe[:,0]) #np.concatenate([resc_stock_np, resc_stock_np, resc_stock_np, resc_stock_np])
arg2 = np.concatenate([selected_simu_np, selected_simu_np, selected_simu_np, selected_simu_np, selected_simu_np])#, selected_simu_np])
arg2 = np.concatenate([selected_simu_np, selected_simu_np])

bins = np.histogram_bin_edges(np.concatenate([arg, arg2]), bins=50)

bins2 = []
for i in range(0,400,16):
    bins2.append(i/2)

print(len(arg), len(arg2))

axes[0].hist(arg, bins=bins2, density=False, histtype='step', color='black', alpha=1, linewidth = 4, label='Generated')
axes[0].hist(arg2, bins=bins2, density=False, facecolor='b', alpha=0.8, label='Input')
axes[0].set_xlabel("MuonsPt")
axes[0].set_ylabel(r'Number of events', fontsize=12)
axes[0].grid(True)
axes[0].set_title("Generated vs Input (Epochs: {})".format(epochs))
axes[0].set_xlim(0, 170)
axes[0].legend()

arg1 = np.squeeze(resc_stock_safe[:,1]) #np.concatenate([resc_stock_np, resc_stock_np, resc_stock_np, resc_stock_np])
arg21 = np.concatenate([selected_simu_np1, selected_simu_np1, selected_simu_np1, selected_simu_np1, selected_simu_np1])#, selected_simu_np1])
arg21 = np.concatenate([selected_simu_np1, selected_simu_np1])

bins1 = np.histogram_bin_edges(np.concatenate([arg1, arg21]), bins=50)

print(len(arg1), len(arg21))

axes[1].hist(arg1, bins=bins2, density=False, histtype='step', color='black', alpha=1, linewidth = 4, label='Generated')
axes[1].hist(arg21, bins=bins2, density=False, facecolor='b', alpha=0.8, label='Input')
axes[1].set_xlabel("invMass")
# axes[1].set_ylabel(r'$Number\ of\ events \times 2$', fontsize=12)
axes[1].grid(True)
#axes[1].set_title("Generated vs Input (Epochs: {})".format(epochs))
axes[1].set_xlim(50, 200)
axes[1].legend()

arg11 = np.squeeze(resc_stock_safe[:,2]) #np.concatenate([resc_stock_np, resc_stock_np, resc_stock_np, resc_stock_np])
arg211 = np.concatenate([selected_simu_np2, selected_simu_np2, selected_simu_np2, selected_simu_np2, selected_simu_np2])#, selected_simu_np2])
arg211 = np.concatenate([selected_simu_np2, selected_simu_np2])

bins11 = np.histogram_bin_edges(np.concatenate([arg11, arg211]), bins=50)

print(len(arg11), len(arg211))

axes[2].hist(arg11, bins=bins2, density=False, histtype='step', color='black', alpha=1, linewidth = 4, label='Generated')
axes[2].hist(arg211, bins=bins2, density=False, facecolor='b', alpha=0.8, label='Input')
axes[2].set_xlabel("MET_pt")
# axes[2].set_ylabel(r'$Number\ of\ events \times 2$', fontsize=12)
axes[2].grid(True)
#axes[1].set_title("Generated vs Input (Epochs: {})".format(epochs))
axes[2].set_xlim(0, 150)
axes[2].legend()

plt.show()
#%%

bins = np.histogram_bin_edges(np.concatenate([arg, arg2]), bins=50)

plt.hist(arg, bins=bins, density=False, histtype='step', color='black', alpha=1, linewidth=3, label='Generated')
plt.hist(arg2, bins=bins, density=False, facecolor='b', alpha=0.8, label='Input')
plt.xlabel("GeV/c", fontsize = 20)
plt.ylabel('Number of events', fontsize=20)
plt.grid(True)
# plt.title("Generated vs Input (Epochs: {})".format(epochs))
plt.title("Generated vs Input : MuonPt", fontsize=20)
plt.legend()
plt.tick_params(axis='both', which='major', labelsize=15)

plt.show()

#%%
bins1 = np.histogram_bin_edges(np.concatenate([arg1, arg21]), bins=50)

plt.hist(arg1, bins=bins1, density=False, histtype='step', color='black', alpha=1, linewidth=3, label='Generated')
plt.hist(arg21, bins=bins1, density=False, facecolor='b', alpha=0.8, label='Input')
plt.xlabel("GeV/c²", fontsize = 20)
plt.ylabel('Number of events', fontsize=20)
plt.grid(True)
plt.title("Generated vs Input : invMass", fontsize = 20)
plt.legend()
plt.tick_params(axis='both', which='major', labelsize=15)

plt.show()

#%%

bins1 = np.histogram_bin_edges(np.concatenate([arg11, arg211]), bins=50)

plt.hist(arg11, bins=bins1, density=False, histtype='step', color='black', alpha=1, linewidth=3, label='Generated')
plt.hist(arg211, bins=bins1, density=False, facecolor='b', alpha=0.8, label='Input')
plt.xlabel("GeV/c", fontsize = 20)
plt.ylabel('Number of events', fontsize=20)
plt.grid(True)
plt.title("Generated vs Input : MET_pt", fontsize=20)
plt.legend()
plt.tick_params(axis='both', which='major', labelsize=15)

plt.show()
#%%
# Create subplots for each pair of variables
fig, axs = plt.subplots(1, 3, figsize=(15, 5))

# Plot 2D histogram for the first pair of variables
axs[0].hist2d(arg, arg1, bins=(80, 80), range=[[0, 80], [0, 125]], cmap=plt.cm.rainbow, alpha = 1)
#•axs[0].hist2d(selected_simu_np, selected_simu_np1, bins=(80, 80), range=[[0, 100], [0, 200]], cmap='Blues', alpha = 0.5)
axs[0].set_xlabel(keys_f[0])
axs[0].set_ylabel(keys_f[1])
fig.colorbar(axs[0].collections[0], ax=axs[0])

# Plot 2D histogram for the second pair of variables
axs[1].hist2d(arg, arg11, bins=(100, 100), range=[[0, 80], [0, 30]], cmap=plt.cm.rainbow, alpha = 1)
axs[1].set_xlabel(keys_f[0])
axs[1].set_ylabel(keys_f[2])
fig.colorbar(axs[1].collections[0], ax=axs[1])


# Plot 2D histogram for the third pair of variables
axs[2].hist2d(arg1, arg11, bins=(100, 100), range=[[70, 110], [0, 30]], cmap=plt.cm.rainbow, alpha = 1)
axs[2].set_xlabel(keys_f[1])
axs[2].set_ylabel(keys_f[2])
fig.colorbar(axs[2].collections[0], ax=axs[2])

plt.tight_layout()
plt.show()
#%%
fig, axs = plt.subplots(1, 2, figsize=(12, 6))

# Plot contour for the first pair of variables
axs[0].hist2d(arg, arg1, bins=(80, 80), range=[[0, 80], [0, 125]], cmap=plt.cm.rainbow, alpha = 1)
axs[0].set_xlabel(keys_f[0], fontsize = 12)
axs[0].set_ylabel(keys_f[1], fontsize = 12)
axs[0].set_title("Outputs", fontsize = 15)
fig.colorbar(axs[0].collections[0], ax=axs[0])

axs[1].hist2d(arg2, arg21, bins=(80, 80), range=[[0, 80], [0, 125]], cmap=plt.cm.rainbow, alpha = 1)
axs[1].set_xlabel(keys_f[0], fontsize = 12)
axs[1].set_ylabel(keys_f[1], fontsize = 12)
axs[1].set_title("Inputs", fontsize = 15)
fig.colorbar(axs[1].collections[0], ax=axs[1])
#%%
fig, axs = plt.subplots(1, 2, figsize=(12, 6))

# Plot contour for the first pair of variables
axs[0].hist2d(arg, arg11, bins=(80, 80), range=[[0, 80], [0, 30]], cmap=plt.cm.rainbow, alpha = 1)
axs[0].set_xlabel(keys_f[0], fontsize = 12)
axs[0].set_ylabel(keys_f[1], fontsize = 12)
axs[0].set_title("Outputs", fontsize = 15)
fig.colorbar(axs[0].collections[0], ax=axs[0])

axs[1].hist2d(arg2, arg211, bins=(100, 100), range=[[0, 80], [0, 30]], cmap=plt.cm.rainbow, alpha = 1)
axs[1].set_xlabel(keys_f[0], fontsize = 12)
axs[1].set_ylabel(keys_f[2], fontsize = 12)
axs[1].set_title("Inputs", fontsize = 15)
fig.colorbar(axs[1].collections[0], ax=axs[1])
#%%
fig, axs = plt.subplots(1, 2, figsize=(12, 6))

# Plot contour for the first pair of variables
axs[0].hist2d(arg1, arg11, bins=(80, 80), range=[[50, 130], [0, 30]], cmap=plt.cm.rainbow, alpha = 1)
axs[0].set_xlabel(keys_f[1], fontsize = 12)
axs[0].set_ylabel(keys_f[2], fontsize = 12)
axs[0].set_title("Outputs", fontsize = 15)
fig.colorbar(axs[0].collections[0], ax=axs[0])

axs[1].hist2d(arg21, arg211, bins=(100, 100), range=[[50, 130], [0, 30]], cmap=plt.cm.rainbow, alpha = 1)
axs[1].set_xlabel(keys_f[1], fontsize = 12)
axs[1].set_ylabel(keys_f[2], fontsize = 12)
axs[1].set_title("Inputs", fontsize = 15)
fig.colorbar(axs[1].collections[0], ax=axs[1])
#%%
from scipy.stats import ks_2samp

# Define the pairs of distributions
pairs = [(arg, arg2), (arg1, arg21), (arg11, arg211)]

# Perform Kolmogorov-Smirnov test
print("Kolmogorov-Smirnov Test:")
for i, (dist1, dist2) in enumerate(pairs, start=1):
    ks_stat, ks_pval = ks_2samp(dist1, dist2)
    print(f"Pair {i}: KS statistic = {ks_stat:.10f}, p-value = {ks_pval:.10f}")

#%%

hist1, bins1, _ = plt.hist(resc_stock_safe[:,0], bins=50, alpha=0.5, label='Ouputs')
hist2, bins2, _ = plt.hist(selected_simu_np, bins=50, alpha=0.5, label='Inputs')

# Normalize histograms
hist1_normalized = hist1 / np.sum(hist1)
hist2_normalized = hist2 / np.sum(hist2)

# Plot normalized histograms
plt.clf()  # Clear previous plot
plt.bar(bins2[:-1], hist2_normalized, width=np.diff(bins1), alpha=1, color='blue', label='Inputs')
plt.bar(bins1[:-1], hist1_normalized, width=np.diff(bins2), alpha=1, color='none', edgecolor='black', linewidth=3, label='Outputs')


# Add labels and legend
plt.xlabel('Value')
plt.ylabel('Normalized Frequency')
plt.title('Comparison of Two Samples')
plt.legend()

# Show plot
plt.show()

hist1, bins1, _ = plt.hist(resc_stock_safe[:,1], bins=50, alpha=0.5, label='Ouputs')
hist2, bins2, _ = plt.hist(selected_simu_np1, bins=50, alpha=0.5, label='Inputs')

# Normalize histograms
hist1_normalized = hist1 / np.sum(hist1)
hist2_normalized = hist2 / np.sum(hist2)

# Plot normalized histograms
plt.clf()  # Clear previous plot
plt.bar(bins2[:-1], hist2_normalized, width=np.diff(bins1), alpha=1, color='blue', label='Inputs')
plt.bar(bins1[:-1], hist1_normalized, width=np.diff(bins2), alpha=1, color='none', edgecolor='black', linewidth=3, label='Outputs')


# Add labels and legend
plt.xlabel('Value')
plt.ylabel('Normalized Frequency')
plt.title('Comparison of Two Samples')
plt.legend()

# Show plot
plt.show()

hist1, bins1, _ = plt.hist(resc_stock_safe[:,2], bins=50, alpha=0.5, label='Ouputs')
hist2, bins2, _ = plt.hist(selected_simu_np2, bins=50, alpha=0.5, label='Inputs')

# Normalize histograms
hist1_normalized = hist1 / np.sum(hist1)
hist2_normalized = hist2 / np.sum(hist2)

# Plot normalized histograms
plt.clf()  # Clear previous plot
plt.bar(bins2[:-1], hist2_normalized, width=np.diff(bins1), alpha=1, color='blue', label='Inputs')
plt.bar(bins1[:-1], hist1_normalized, width=np.diff(bins2), alpha=1, color='none', edgecolor='black', linewidth=3, label='Outputs')


# Add labels and legend
plt.xlabel('Value')
plt.ylabel('Normalized Frequency')
plt.title('Comparison of Two Samples')
plt.legend()

# Show plot
plt.show()
#%%
sample1 = resc_stock_safe[:,0]
sample2 = selected_simu_np

num_bins = 30
bin_width = 20 / num_bins

# Create histograms with fixed number of bins
hist1, bins1 = np.histogram(sample1, bins=num_bins)
hist2, bins2 = np.histogram(sample2, bins=num_bins)

# Normalize histograms by dividing by the total number of events multiplied by the bin width
hist1_normalized = hist1 / (len(sample1) * bin_width)
hist2_normalized = hist2 / (len(sample2) * bin_width)

# Plot normalized histograms
plt.bar(bins1[:-1], hist1_normalized, width=bin_width, alpha=0.5, label='Sample 1')
plt.bar(bins2[:-1], hist2_normalized, width=bin_width, alpha=0.5, label='Sample 2')

# Add labels and legend
plt.xlabel('Value')
plt.ylabel('Normalized Frequency')
plt.title('Comparison of Normalized Distributions')
plt.legend()

# Show plot
plt.show()

#%%

pairs = [(hist1_normalized, hist2_normalized)]

# Perform Kolmogorov-Smirnov test
print("Kolmogorov-Smirnov Test:")
for i, (dist1, dist2) in enumerate(pairs, start=1):
    ks_stat, ks_pval = ks_2samp(dist1, dist2)
    print(f"Pair {i}: KS statistic = {ks_stat:.10f}, p-value = {ks_pval:.10f}")
    
#%%
sample_pairs = [
    (resc_stock_safe[:, 0], selected_simu_np),
    (resc_stock_safe[:, 1], selected_simu_np1),
    (resc_stock_safe[:, 2], selected_simu_np2)
]

num_bins = 30
bin_width = 100 / num_bins

pairs = []
pairs_raw = []

# Loop over sample pairs and create separate plots for each pair
for i, (sample1, sample2) in enumerate(sample_pairs):
    # Create histograms with fixed number of bins
    hist1, bins1 = np.histogram(sample1, bins=num_bins)
    hist2, bins2 = np.histogram(sample2, bins=num_bins)
    
    pairs_raw.append((hist1, hist2))
    
    # Normalize histograms by dividing by the total number of events multiplied by the bin width
    hist1_normalized = hist1 / (len(sample1) * bin_width)
    hist2_normalized = hist2 / (len(sample2) * bin_width)
    
    # Store the normalized histograms in pairs
    pairs.append((hist1_normalized, hist2_normalized))
    
    # plt.figure(figsize=(12, 8))
    
    # plt.bar(bins1[:-1], hist1_normalized, width=bin_width, alpha=1, color='none', edgecolor='black', linewidth=4, label='Output')
    # plt.bar(bins2[:-1], hist2_normalized, width=bin_width, alpha=1, color='blue', edgecolor='black', label='Input')

    # plt.xlabel('Value')
    # plt.ylabel('Normalized Frequency')
    # plt.title(f'Comparison of Normalized Distributions - Pair {i+1}')
    # plt.legend()
    
    # plt.tick_params(axis='both', which='major', labelsize=15)
    
    # plt.show()
    
#%%

# Perform the KS test on each pair of normalized histograms
for i, (dist1, dist2) in enumerate(pairs, start=1):
    ks_stat, ks_pval = ks_2samp(dist1, dist2)
    print(f"Pair {i}: KS statistic = {ks_stat:.10f}, p-value = {ks_pval:.10f}")
    
#%%
from scipy.stats import chi2_contingency

for i, (dist1, dist2) in enumerate(pairs_raw, start=1):
    chi2_stat, p_val, dof, ex = chi2_contingency([dist1, dist2])
    print(f"Pair {i}: Chi-squared statistic = {chi2_stat:.10f}, p-value = {p_val:.10f}")
    
#%%
import warnings
from sklearn.metrics import mutual_info_score
from sklearn.metrics import normalized_mutual_info_score

warnings.filterwarnings("ignore")

mi_arg_arg1 = mutual_info_score(arg, arg1)
mi_arg_arg11 = mutual_info_score(arg, arg11)
mi_arg1_arg11 = mutual_info_score(arg1, arg11)

nmi_arg_arg1 = normalized_mutual_info_score(arg, arg1)
nmi_arg_arg11 = normalized_mutual_info_score(arg, arg11)
nmi_arg1_arg11 = normalized_mutual_info_score(arg1, arg11)

# Calculate mutual information and normalized mutual information for Set 2
mi_arg2_arg21 = mutual_info_score(selected_simu_np, selected_simu_np1)
mi_arg2_arg211 = mutual_info_score(selected_simu_np, selected_simu_np2)
mi_arg21_arg211 = mutual_info_score(selected_simu_np1, selected_simu_np2)

nmi_arg2_arg21 = normalized_mutual_info_score(selected_simu_np, selected_simu_np1)
nmi_arg2_arg211 = normalized_mutual_info_score(selected_simu_np, selected_simu_np2)
nmi_arg21_arg211 = normalized_mutual_info_score(selected_simu_np1, selected_simu_np2)

# Compare mutual information values
# Compare mutual information values
print("Comparing mutual information values : Outputs vs Inputs")
print("MI(MuonPt, invMass) : ", mi_arg_arg1, "vs", mi_arg2_arg21)
print("NMI(MuonPt, invMass) : ", nmi_arg_arg1, "vs", nmi_arg2_arg21)

print("MI(invMass, MET_pt) : ", mi_arg1_arg11, "vs", mi_arg21_arg211)
print("NMI(invMass, MET_pt) : ", nmi_arg1_arg11, "vs", nmi_arg21_arg211)

print("MI(MuonPt, MET_pt) : ", mi_arg_arg11, "vs", mi_arg2_arg211)
print("NMI(MuonPt, MET_pt) : ", nmi_arg_arg11, "vs", nmi_arg2_arg211)

#%%

import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_regression

# Convert to pandas Series if they are not already
arg1_series = pd.Series(arg1, name='arg1')
selected_simu_np_series = pd.Series(selected_simu_np, name='selected_simu_np')

# Combine into a DataFrame
data = pd.DataFrame({'arg1': arg1_series, 'selected_simu_np': selected_simu_np_series})

#%%

# Calculate mutual information
num_iterations = 100
mi_sum = 0

for _ in range(num_iterations):
    mi = mutual_info_regression(data[['arg1']][:3000], data['selected_simu_np'][:3000])[0]
    mi_sum += mi

mi_avg = mi_sum / num_iterations

print("Mutual information mean:")
print(mi_avg)

#%%
from copulas.multivariate import GaussianMultivariate
from copulas.visualization import scatter_2d

# Create dataframes for easier handling
df1 = pd.DataFrame({'arg': arg, 'arg1': arg1, 'arg11': arg11})
df2 = pd.DataFrame({'selected_simu_np': selected_simu_np, 'selected_simu_np1': selected_simu_np1, 'selected_simu_np2': selected_simu_np2})

# Sample 5000 events from each dataframe
sampled_df1 = df1.sample(n=5000, random_state=1)
sampled_df2 = df2.sample(n=5000, random_state=1)

# Fit Gaussian Copula models
copula1 = GaussianMultivariate()
copula1.fit(sampled_df1)

copula2 = GaussianMultivariate()
copula2.fit(sampled_df2)

# Generate samples from the copula models
sampled_df1 = copula1.sample(5000)
sampled_df2 = copula2.sample(5000)

# Visualize the Copula dependencies
scatter_2d(sampled_df1, ['arg', 'arg1'])
plt.show()
scatter_2d(sampled_df1, ['arg1', 'arg11'])
plt.show()
scatter_2d(sampled_df1, ['arg', 'arg11'])
plt.show()

scatter_2d(sampled_df2, ['selected_simu_np', 'selected_simu_np1'])
plt.show()
scatter_2d(sampled_df2, ['selected_simu_np1', 'selected_simu_np2'])
plt.show()
scatter_2d(sampled_df2, ['selected_simu_np', 'selected_simu_np2'])
plt.show()

