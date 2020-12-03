#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from numpy.linalg import inv
from numpy import linalg 
import csv
import math
import random


# In[2]:


def softmax(x):
    x_2 = x - np.max(x)
    return (np.exp(x_2).T / np.sum(np.exp(x_2),axis = 1)).T

def get_y(data,n,k,w):    
    a_k = w[k].T.dot(data[n].reshape(-1,1))   
    s = 0    
    for j in range(5):
        a_j = w[j].T.dot(data[n].reshape(-1,1))
        s += (np.exp(a_j - a_k))
    s = 1/s
    
    return s

def get_pred(data,w,n_class):
    probability_max = 0
    max_label = -1
    for i in range(n_class):
        probability = 0
        s = 0
        a_i = w[i].T.dot(data.reshape(-1,1))   
        for j in range(n_class):
            a_j = w[j].T.dot(data.reshape(-1,1))   
            s += np.exp(a_i - a_j)
        probability = 1/s
        if(probability > 0.5):
            return i
        if(probability > probability_max):
            probability_max = probability
            max_label = i

    return max_label

def get_err(data, w, y, allk):   
    s = 0
    for k in range(allk):
        for n in range(data.shape[0]):
            if(y[:,k][n] != 0):
                s += np.log(get_y(data,n,k,w))
    s *= (-1)
    
    return s

def get_hess(data, w ,k):
    matr = np.zeros((w.shape[1],w.shape[1]))
    for n in range(data.shape[0]):
        s = get_y(data,n,k,w) *  (1 - get_y(data,n,k,w))
        hereX = data[n]
        matr += (float(s) * np.outer(hereX,hereX))
    return matr

def get_grad(data, w, y, k):  
    re = np.zeros((w.shape[1],1))
    for n in range(data.shape[0]):
        s = get_y(data,n,k,w) - y[:,k][n]
        hereX = data[n].reshape(-1,1)
        re += float(s) * hereX
    return re

def GD(trainData,trainLabel,trainLabel_oneHot,testData,testLabel,testLabel_oneHot,gdType,PCA_d = -1): 
    
    if(PCA_d > 0):
        
        trainData, meanInTrain, pcaSpace = PCA(trainData,PCA_d)
    
        newTestData = testData - meanInTrain    
        out = newTestData.dot(pcaSpace)
        testData = out.astype('float64')
     
    batchSize = -1
    w = np.zeros((trainData.shape[1],n_class))
    learningRate = 0.001
    epochs = 1000   
    N = len(trainData)
    
    if(gdType == "batch GD"):
        batchSize = N
    elif(gdType == "SGD"):
        batchSize = 1
    elif(gdType == "mini-batch SGD"):
        batchSize = 32
    else:
        print("[INFO] Wrong INPUT")
        return
    
    trainAcc = []
    testAcc = []
    trainLoss = []
    testLoss = []
    
    for epoch in range(epochs):       
        iterations = int(N/batchSize)
        thisRandomData = np.copy(trainData)
        thisRandomLabel = np.copy(trainLabel_oneHot)
        c1 = list(zip(thisRandomData,thisRandomLabel))
        random.shuffle(c1)
        thisRandomData,thisRandomLabel = zip(*c1) 
        thisLoss = -1
        for oneBatch in range(iterations):
            nowX = np.array(thisRandomData[oneBatch*batchSize:oneBatch*batchSize+batchSize])
            nowY = np.array(thisRandomLabel[oneBatch*batchSize:oneBatch*batchSize+batchSize])

            output0 = nowX.dot(w)
            output0 = softmax(output0)
            thisLoss = (-1 * np.sum(nowY * np.log(output0))) / batchSize
            
            gradient = -1 * nowX.T.dot(nowY - output0) / batchSize
            w -= learningRate * gradient
            
        
        prediction = softmax(trainData.dot(w))
        thisLoss = (-1 * np.sum(trainLabel_oneHot * np.log(prediction))) / len(trainData)
        trainLoss.append(thisLoss)
        prediction = np.argmax(prediction,axis = 1)
        right = 0
        for oneSample in range(N):
            if(prediction[oneSample]==trainLabel[oneSample]):
                right += 1
        trainAcc.append(right/N)   
        
        prediction = softmax(testData.dot(w))
        thisLoss = (-1 * np.sum(testLabel_oneHot * np.log(prediction))) / len(testData)
        testLoss.append(thisLoss)
        prediction = np.argmax(prediction,axis = 1)
        right = 0
        for oneSample in range(len(testData)):
            if(prediction[oneSample]==testLabel[oneSample]):
                right += 1
        testAcc.append(right/len(testData))
        
    plt.figure()    
    plt.title(gdType + " _Accuracy")
    plt.plot(trainAcc,label='train')
    plt.plot(testAcc,label='test')
    plt.legend()
    plt.show
    
    plt.figure()    
    plt.title(gdType + " _Loss")
    plt.plot(trainLoss,label='train')
    plt.plot(testLoss,label='test')
    plt.legend()
    plt.show
    
    
    print("----------------------------")
    print("TYPE :",gdType)
    if(PCA_d == -1):
        print("no PCA")
    else:
        print("using PCA, d =",PCA_d)
    print("training Accuracy :",trainAcc[len(trainAcc)-1])
    print("test Accuracy :",testAcc[len(testAcc)-1])
       
    return


def Raphson_Newton(trainData,trainLabel,trainLabel_oneHot,testData,testLabel,testLabel_oneHot,gdType,PCA_d = 10):
    
    if(PCA_d > 0):
        
        trainData, meanInTrain, pcaSpace = PCA(trainData,PCA_d)
    
        newTestData = testData - meanInTrain    
        out = newTestData.dot(pcaSpace)
        testData = out.astype('float64')
    print(trainData.shape)
    print(trainLabel.shape)
    trainAcc = []
    testAcc = []
    trainLoss = []
    testLoss = []
    
    w = np.zeros((n_class, trainData.shape[1], 1))
    epochs = 10
    err = 0
    for epoch in range(epochs):
        err += get_err(trainData, w, trainLabel_oneHot, n_class)
        for k in range(n_class):
            w[k] -= inv(get_hess(trainData, w ,k)).dot(get_grad(trainData, w, trainLabel_oneHot, k))
            print("gre:", get_grad(trainData, w, trainLabel_oneHot, k))
        if(epoch>=0):
            right = 0
            for row in range(len(trainLabel)):
                if(trainLabel[row] == get_pred(trainData[row],w,n_class)):
                    right += 1

                    
            print(right,"g")
            trainAcc.append(right/trainData.shape[0])
        
    plt.figure()    
    plt.title(gdType + " _Accuracy")
    plt.plot(trainAcc,label='train')
    #plt.plot(testAcc,label='test')
    plt.legend()
    plt.show()
    
def PCA(data, d):
    data_mean = np.mean(data,axis = 0)
    newData = data - data_mean
    
    covM = np.cov(newData,rowvar = False)
    eig_value,eig_vector = np.linalg.eig(np.array(covM))
    
    dEigen = eig_vector[:,np.argsort(eig_value)[-1:d * (-1) -1:-1]]
    newVector = newData.dot(dEigen)
    
    w = 28
    h = 28
    
    for i in range(dEigen.T.shape[0]):
        print("eigenvector ", i+1)
        plt.imshow(np.array(dEigen.T[i].reshape(h,w),dtype = 'float32'),cmap = 'gray')        
        plt.show()
    newVector = np.asarray(newVector)
    
    return newVector.astype('float32'),data_mean,dEigen


# In[3]:


n_class = 5
n_sampleEachClass = 64

trainDatas = []
trainLabels = []
testDatas = []
testLabels = []
for labell in range(n_class):
    randomIndex = np.random.permutation(np.arange(n_sampleEachClass))[:n_sampleEachClass//2]
    for i in range(n_sampleEachClass):
        img = mpimg.imread('fashion_mnist_'+ str(labell) + '/' + str(i) + '.png')
        img = img.flatten()
        if(i in randomIndex):
            trainDatas.append(img)
            trainLabels.append(labell)
        else:
            testDatas.append(img)
            testLabels.append(labell)
        

c1 = list(zip(trainDatas,trainLabels))
random.shuffle(c1)
trainDatas,trainLabels = zip(*c1)

c2 = list(zip(testDatas,testLabels))
random.shuffle(c2)
testDatas,testLabels = zip(*c2)

trainDatas = np.array(trainDatas)
trainLabels = np.array(trainLabels)
testDatas = np.array(testDatas)
testLabels = np.array(testLabels)

n_values = np.max(trainLabels) + 1
trainLabel_oneHots = np.eye(n_values)[trainLabels]
n_values = np.max(testLabels) + 1
testLabel_oneHots = np.eye(n_values)[testLabels]


# In[4]:


#PCA(trainData,5)
#GD(PCA(trainData,10),trainLabel,trainLabel_oneHot,PCA(testData,10),testLabel,testLabel_oneHot,"batch GD")

Raphson_Newton(trainDatas,trainLabels,trainLabel_oneHots,testDatas,testLabels,testLabel_oneHots,"SGD",5)
#GD(trainDatas,trainLabels,trainLabel_oneHots,testDatas,testLabels,testLabel_oneHots,"SGD",5)
#GD(trainData,trainLabel,trainLabel_oneHot,testData,testLabel,testLabel_oneHot,"SGD")

