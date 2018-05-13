# -*- coding: utf-8 -*-
"""
Created on Tue Nov 14 19:46:05 2017

@author: Rifayat Samee(Sanzee)
"""

import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import Isomap
import matplotlib.pyplot as plt
import PIL.Image
import os
import sys
from scipy import ndimage
import pickle
from mpl_toolkits.mplot3d import Axes3D
image_size = 50
num_class = 50
ch = 1
def make_the_data_ready_for_Isomap(Data,Labels):
    Data = Data.reshape(-1,image_size*image_size).astype(np.float32)
    #Labels = (np.arange(num_class) == Labels[:,None]).astype(np.float32)
    return Data,Labels
def get_datasets():
    with open("Processed_dataset/datasets.pickle",'rb') as f:
        Data = pickle.load(f)
        train_data = Data['train_data']
        train_label = Data['train_label']
        valid_data = Data['validation_data']
        valid_label = Data['validation_label']
        test_data = Data['test_data']
        test_label = Data['test_label']
    
    return train_data,train_label,valid_data,valid_label,test_data,test_label

train_data,train_label,valid_data,valid_label,test_data,test_label = get_datasets()

train_data,train_label = make_the_data_ready_for_Isomap(train_data,train_label)
valid_data,valid_label = make_the_data_ready_for_Isomap(valid_data,valid_label)
test_data,test_label = make_the_data_ready_for_Isomap(test_data,test_label)

print(train_data.shape)
print(train_label.shape)

# lets make the sample dataset for ISOmap
ISO_data = np.ndarray(shape=[500,2500])
ISO_labels = np.ndarray(shape=[500])
index = 0
for i in range(50):
    t = 0
    for x in range(train_data.shape[0]):
        if(t == 10):
            break
        if(train_label[x] == i):
            ISO_data[index] = train_data[x]
            ISO_labels[index] = train_label[x]
            t = t + 1
            index = index + 1 

print("trying ISOMAP on training data.....")
iso = Isomap(n_components=3)

trainsform = iso.fit_transform(X=ISO_data)
#plt.figure(figsize=(10,10))
#plt.scatter(trainsform[:,0],trainsform[:,1],trainsform[:,2],c=ISO_labels,cmap=plt.cm.get_cmap('jet', 50))
#plt.colorbar(ticks=range(50))

fig = plt.figure(figsize=(10,10))
ax = fig.add_subplot(111, projection = '3d')
ax.scatter(trainsform[:,0],trainsform[:,1],trainsform[:,2],c=ISO_labels,cmap=plt.cm.get_cmap('jet', 50))
#plt.colorbar(ticks=range(50))

