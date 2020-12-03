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
def phi(x):
    return x.reshape(len(x), 1)

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

def hessian(w, k, X):

    output = np.zeros((len(w[0]), len(w[0])))
    for n in range(len(X)):
        scale = compute_y(n, k, w, X) * (1 - compute_y(n, k, w, X))
        output += scale * (phi(X[n]).dot(phi(X[n]).T))
    return output

def compute_y(n, k, w, X):  # 公式裡的y
    s = np.float64(0.)
    ak = w[k].T.dot(phi(X[n]))
    # target classes
    for j in range(5):
        aj = w[j].T.dot(phi(X[n]))
        s += np.nan_to_num(np.exp(aj - ak))
    s = np.nan_to_num(s)

    return 1. / s

def gradient(w, k, t, X):
    output = np.zeros((len(w[0]), 1))
    for n in range(len(X)):
        scale = compute_y(n, k, w, X) - t[:, k][n]  # Ynk - Tnk

        output += scale * phi(X[n])

    return output

def get_pred(dataX,w,n_class):
    rePrediction = []
    for data in dataX:
        probability_max = 0
        max_label = -1
        tag = 1
        for i in range(n_class):
            if(tag):
                probability = 0
                s = 0
                a_i = w[i].T.dot(data.reshape(-1,1))   
                for j in range(n_class):
                    a_j = w[j].T.dot(data.reshape(-1,1))   
                    s += np.exp(a_j - a_i)
                probability = 1/s
                if(probability>0.5):
                    max_label = i
                    tag = 0
                elif(probability > probability_max):
                    probability_max = probability
                    max_label = i
        rePrediction.append(max_label)
    return np.array(rePrediction)

def get_err(data, w, y, allk):   
    s = 0
    for k in range(allk):
        for n in range(data.shape[0]):
            if(y[:,k][n] != 0):
                s += np.log(get_y(data,n,k,w))
    s *= (-1)
    
    return s

def get_hess(data, w ,k):
    return hessian(w, k, data)
    #matr = np.zeros((w.shape[1],w.shape[1]))
    #for n in range(data.shape[0]):
    #    s = get_y(data,n,k,w) *  (1 - get_y(data,n,k,w))
    #    hereX = data[n]
    #    matr += (float(s) * np.outer(hereX,hereX))
    #return matr

def get_grad(data, w, y, k):  
    return  gradient(w, k, y, data)
    #re = np.zeros((w.shape[1],1))
    #for n in range(data.shape[0]):
    #    s = get_y(data,n,k,w) - y[:,k][n]
    #    hereX = data[n].reshape(-1,1)
    #    re += float(s) * hereX
    #return re

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

def predicts(w, x, classes):
    softmaxes = []
    for k in range(classes):
        s = np.float64(0.)
        ak = w[k].T.dot(phi(x))
        for j in range(classes):
            aj = w[j].T.dot(phi(x))
            s += np.nan_to_num(np.exp(aj - ak))
        softmaxes += [1./s]
    return np.where(np.array(softmaxes).reshape(-1) > 1/ classes, 1, 0)
def Raphson_Newton(trainData,trainLabel,trainLabel_oneHot,testData,testLabel,testLabel_oneHot,gdType,PCA_d = 10):
    
    if(PCA_d > 0):
        
        trainData, meanInTrain, pcaSpace = PCA(trainData,PCA_d)
    
        newTestData = testData - meanInTrain    
        out = newTestData.dot(pcaSpace)
        testData = out.astype('float64')

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
        prediction = []

        if(epoch>=0):
            right = 0
            for row in range(len(trainLabel)):
                if(trainLabel[row] == get_pred(trainData,w,n_class)[row]):
                    right += 1
        
                    
            trainAcc.append(right/trainData.shape[0])
    print(trainAcc)

    x_min, x_max = trainData[:, 0].min() - 1, trainData[:, 0].max() + 1
    y_min, y_max = trainData[:, 1].min() - 1, trainData[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),np.arange(y_min, y_max, 0.02))

    print(np.array([xx.ravel(), yy.ravel()]).T.shape)
    # here "model" is your model's prediction (classification) function

  

    Z = get_pred(np.array([xx.ravel(), yy.ravel()]).T,w,n_class)

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z,alpha= 0.3)

    #for oneX in trainData.shape[0]:
    #    if()
    plt.scatter(trainData[:,0],trainData[:,1],marker = 'o' ,alpha=0.8,c=get_pred(trainData,w,n_class))
    plt.scatter(testData[:,0],testData[:,1],marker = 'x',alpha=0.8,c=testLabel)
    plt.axis('off')
    
    # Plot also the training points
    #plt.scatter(X[:, 0], X[:, 1], c=Y, cmap=pl.cm.Paired)
    #
    #plt.figure()    
    #plt.title(gdType + " _Accuracy")
    #plt.plot(trainAcc,label='train')
    ##plt.plot(testAcc,label='test')
    #plt.legend()
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

Raphson_Newton(trainDatas,trainLabels,trainLabel_oneHots,testDatas,testLabels,testLabel_oneHots,"SGD",2)
#GD(trainDatas,trainLabels,trainLabel_oneHots,testDatas,testLabels,testLabel_oneHots,"SGD",5)
#GD(trainData,trainLabel,trainLabel_oneHot,testData,testLabel,testLabel_oneHot,"SGD")

