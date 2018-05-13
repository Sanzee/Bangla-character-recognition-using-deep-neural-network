#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 10 22:01:14 2018
Convolutional NET for Bangla Hand_written Alphabets
@author: Rifayat Samee (sanzee)
"""
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.optim as optim 
from sklearn.metrics import f1_score, accuracy_score
Batch_size = 60
image_size = 50
conv1_depth = 16
conv2_depth = 16
patch_size_1 = 3
patch_size_2 = 5
fc1_size = 400
fc2_size = 200
n_class = 50
input_ch = 1
max_epoch = 100
learning_rate = 0.001
eps = 1e-12
dropout = 0.50
def calculate_dim_after_conv(input_dim,kernel_size=2,stride=1,padding=0):
    return [((input_dim[0]-kernel_size + 2*padding)//stride) + 1,((input_dim[1]-kernel_size + 2*padding)//stride) + 1]

def Load_Dataset(filename):
    with open(filename,"rb") as f:
        Data = pickle.load(f)
    return Data['train_data'],Data['train_label'],Data['validation_data'],Data['validation_label'],Data['test_data'],Data['test_label']


class CNN(nn.Module):
    def  __init__(self):
        super(CNN,self).__init__()
        self.conv1 = nn.Conv2d(input_ch,conv1_depth,patch_size_1,stride=1, padding=0)
        self.conv2 = nn.Conv2d(conv1_depth,conv2_depth,patch_size_2,stride=1, padding=0)
        self.fc1 = nn.Linear(conv2_depth*10*10,fc1_size)
        self.fc2 = nn.Linear(fc1_size,fc2_size)
        self.fc_drop = nn.Dropout(dropout)
        self.fc3 = nn.Linear(fc2_size,n_class)
        self.conv_drop = nn.Dropout(dropout)
        
    def forward(self,x):
        
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(self.conv_drop(x),[2,2],stride = 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x,[2,2],stride = 2)
        
        x = x.view(-1,conv2_depth*10*10)
        x = self.fc_drop(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        
        return F.log_softmax(self.fc3(x))

train_x,train_y,vcc_x,vcc_y,test_x,test_y = Load_Dataset("Processed_dataset/datasets.pickle")

def random_batch(epoch):
    
    start_index = Batch_size * epoch
    end_index = start_index + 100
    Batch = Train_Data[start_index:end_index]
    np.random.shuffle(Batch)
    batch_x = torch.FloatTensor([x[0] for x in Batch]).unsqueeze(1)
    batch_y = torch.LongTensor([x[1] for x in Batch])
    
    return batch_x,batch_y
    

def Merge_Validation_With_train():
    for i in range(len(vcc_x)):
        Train_Data.append([vcc_x[i],vcc_y[i]])
        


Train_Data = []
for i in range(len(train_x)):
    Train_Data.append([train_x[i],train_y[i]])

ConvNet = CNN()
print(ConvNet)


LossFunc = nn.CrossEntropyLoss()
optimizer = optim.Adam(ConvNet.parameters(),lr=learning_rate,weight_decay=1e-5)

def Validation_Loss():
    V_x = torch.FloatTensor(vcc_x).unsqueeze(1)
    V_y = torch.LongTensor(vcc_y)
    logit = ConvNet(V_x)
    loss = LossFunc(logit,V_y)
    return loss

def test():
    ConvNet.eval()
    T_x = torch.FloatTensor(test_x).unsqueeze(1)
    T_y = torch.LongTensor(test_y)
    pred = ConvNet(T_x).max(1,keepdim=True)[1]
    test_acc = accuracy_score(T_y,pred)
    test_f1 = f1_score(T_y,pred,average='weighted')
    
    print("test accuracy is : {}%".format(test_acc*100))
    print("test F1 score(weighted) : {}%".format(test_f1*100))
    

def train(early_stopping = False, patience  = 5,validation=False):
    total_batch = len(Train_Data)//Batch_size
    cur_v_loss = 100000.0
    wait = 0
    for i in range(max_epoch):
        ConvNet.train()
        print("Running Epoch {}".format(i))
        for e in range(total_batch):
            optimizer.zero_grad()
            batch_x,batch_y = random_batch(e)
            logit = ConvNet(batch_x)
            loss = LossFunc(logit,batch_y)
            loss.backward()
            optimizer.step()
            print("\t>>batch>> {} Loss: {}".format(e,loss))
        if validation:
            v_loss = Validation_Loss()
            print("Validation Loss: {}".format(v_loss))
            if early_stopping:
                if v_loss < cur_v_loss + eps:
                    cur_v_loss = v_loss
                    wait = 0
                else:
                    if wait == patience-1:
                        print("No improvment on validation set for {} consecutive Epochs".format(patience))
                        break
                        wait += 1
#        test()
        np.random.shuffle(Train_Data)


    

if __name__ == "__main__":
    Merge_Validation_With_train()
    train(validation=False)
    test()

    


