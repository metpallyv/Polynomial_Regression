#Implementation of polynomial regression

__author__ = 'Vardhaman'
import sys
import csv
import math
import copy
import time
import numpy as np
from collections import Counter
from numpy.linalg import inv
from numpy import *#genfromtxt
import matplotlib.pyplot as plt

def load_csv(file):
    X = genfromtxt(file, delimiter=",",dtype=str)
    #print(X)
    return (X)

def random_numpy_array(ar):
    np.random.shuffle(ar)
    #print(arr)
    arr = ar
    #print(arr)
    return arr

def generate_set(X,i):
    X = np.array(X,dtype=float)
    if i == 1:
        y_training = X[:, -1]
        X_training = X[:,:-1]
        y_training = y_training.flatten()
        return (X_training,y_training)
    else:
        y_training = X[:,-1]
        X_initial = X[:,:-1]
        #print(X_initial.shape)
        X = X[:,:-1]
        #print X.shape
        a = range(2,i+1)
        for j in a:
            b = np.power(X_initial,j)
            X = np.insert(X,X.shape[1],values=b.flatten(),axis=1)
        #print X.shape
        X = np.insert(X, 0, values=np.ones(X.shape[0]), axis=1)
        #print(X)
        X = np.insert(X,X.shape[1],values=y_training.flatten(),axis=1)

        y_training = X[:, -1]
        X_training = X[:,:-1]
        y_training = y_training.flatten()
        #print(X_training.shape)
        return (X_training,y_training)

def normal_equation(x,y):
    # calculate weight vector with the formula inverse of(x.T* x)*x.T*y
    z = inv(dot(x.transpose(), x))
    theta = dot(dot(z, x.transpose()), y)
    #print(theta.shape)
    return theta

def compute_rmse_sse(x,y,theta):
    m = y.size
    pred = x.dot(theta)
    #print(pred)
    error = pred - y
    sse = error.T.dot(error)/float(m)
    rmse = math.sqrt(sse)
    #print"SSE:",sse,"RMSE:",rmse
    return rmse,sse

if __name__ == "__main__":
    if len(sys.argv) == 3:
        newfile = sys.argv[1]
        testfile = sys.argv[2]
        num_arr2 = load_csv(testfile)
        #attributes,class_name, feature = load_csv(newfile)
        num_arr1 = load_csv(newfile)

        #num_arr1 = random_numpy_array(num_arr1)
        #Divide the data into 10 cross training and test data
        a = range(1,16)
        trainingRMSEValues = []
        testRMSEValues = []
        for i in a:
            training_x,training_y = generate_set(num_arr1,i)
            test_x,test_y = generate_set(num_arr2,i)
        #print (test_x,len(training_x))
            theta = normal_equation(training_x,training_y)
            rmse,see = compute_rmse_sse(training_x,training_y,theta)
            trainingRMSEValues.append(rmse)
            print "RMSE for training for p value of", i,"is",rmse
            rmse,see = compute_rmse_sse(test_x,test_y,theta)
            testRMSEValues.append(rmse)
            print "RMSE for Validation for p value of ", i, "is ",rmse

        #Plot the graph
        plt.suptitle("Sinusoid Polynomial Regression plot")
        plt.plot(a,trainingRMSEValues)
        plt.ylabel("RMSE")
        plt.plot(a,testRMSEValues)
        plt.ylabel("RMSE")
        plt.xlabel("max(p) ")
        fileName = "sinusoidPlot";
        plt.savefig(fileName)
