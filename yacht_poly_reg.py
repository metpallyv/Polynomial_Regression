__author__ = 'Vardhaman'
import sys
import csv
import math
import copy
import time
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from numpy.linalg import inv
from numpy import *#genfromtxt
feature = []

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

def normalize(matrix,sd,me):
    with np.errstate(divide='ignore'):
        a = matrix
        sd_list = []
        mean_list = []
        if me == 0 and sd == 0:
            b = np.apply_along_axis(lambda x: (x-np.mean(x))/float(np.std(x)),0,a)
            tmp = a.shape[1]
            for i in range(tmp):
                sd_list.append(np.std(a[:,i]))
                mean_list.append(np.mean(a[:,i]))
            return b,sd_list,mean_list

        else:
            res = np.empty(shape=[a.shape[0],0])
            for i in range(a.shape[1]):
                col = matrix[:, i]
                mean_val = me[i]
                std_val = sd[i]
                b = np.apply_along_axis(lambda x: (x-mean_val)/float(std_val),0,col)
                res = np.concatenate((res, b), axis=1)
        res = np.nan_to_num(res)
    return res,sd,me

def generate_set(X,i):
    X = np.array(X,dtype=float)
    Y = X[:,-1]
    X_initial = X[:,:-1]
    X = X[:,:-1]
    #print(X_initial.shape)
    a = range(2,i+1)
    for j in a:
        b = np.power(X_initial,j)
        X = np.append(X,b,axis=1)
    #print X.shape
    X = np.insert(X,X.shape[1],values=Y.flatten(),axis=1)
    num_test = round(0.1*(X.shape[0]))
    start = 0
    end = num_test
    test_attri_list =[]
    test_class_names_list =[]
    training_attri_list = []
    training_class_names_list = []
    for i in range(10):
        X_test = X[start:end , :]
        tmp1 = X[:start, :]
        tmp2 = X[end:, :]
        X_training = np.concatenate((tmp1, tmp2), axis=0)
        y_training = X_training[:, -1]
        y_test = X_test[:, -1]
        X_training = X_training[:,:-1]
        X_test = X_test[:,:-1]
        X_training = np.matrix( X_training )
        X_test = np.matrix(X_test)
        X_training_normalized,sd,mean = normalize(X_training,0,0)
        X_training_normalized = np.nan_to_num(X_training_normalized)
        X_test_normalized,sd,mean = normalize(X_test,sd,mean)
        len1 = X_training_normalized.shape[0]
        len2 = X_test_normalized.shape[0]
        x_training_ones = np.ones(len1)
        x_training_ones = x_training_ones.reshape([x_training_ones.shape[0],1])
        x_test_ones = np.ones(len2)
        x_test_ones = x_test_ones.reshape([x_test_ones.shape[0],1])
        X_training_normalized = np.concatenate((x_training_ones,X_training_normalized),axis=1)
        X_test_normalized = np.concatenate((x_test_ones,X_test_normalized),axis=1)
        y_test = y_test.flatten()
        y_training = y_training.flatten()
        test_attri_list.append(X_test_normalized)
        test_class_names_list.append(y_test)
        training_attri_list.append(X_training_normalized)
        training_class_names_list.append(y_training)
        start = end
        end = end+num_test

    return test_attri_list,test_class_names_list,training_attri_list,training_class_names_list

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
    return rmse,sse

if __name__ == "__main__":
    if len(sys.argv) == 2:
        newfile = sys.argv[1]
        num_arr = load_csv(newfile)
        num_arr = random_numpy_array(num_arr)
        a = range(2,7)
        trainingRMSEValues = []
        testRMSEValues = []
        for p in a:
            #Divide the data into 10 cross training and test data
            test_x,test_y,training_x,training_y = generate_set(num_arr,p)
            #Apply normal form equation for all 10 cross data
            rmse_training_list = []
            rmse_test_list = []
            for i in range(10):
                theta = normal_equation(training_x[i],training_y[i])
                #calculate the rmse and sse for each fold
                rmse1,sse1 = compute_rmse_sse(test_x[i],test_y[i],theta)
                #trainingRMSEValues.append(rmse1)
                rmse2,see2 = compute_rmse_sse(training_x[i],training_y[i],theta)
                #testRMSEValues.append(rmse2)
                rmse_training_list.append(rmse2)
                rmse_test_list.append(rmse1)

            meanTrainingRMSE = sum(rmse_training_list)/float(10)
            print "Mean RMSE for training p",p,"is",meanTrainingRMSE
            trainingRMSEValues.append(meanTrainingRMSE)
            meanTestingRMSE = sum(rmse_test_list)/float(10)
            testRMSEValues.append(meanTestingRMSE)
            print "Mean RMSE for test p",p,"is",meanTestingRMSE

        plt.suptitle("Yacht Ploynomial Regression plot")
        plt.plot(a,trainingRMSEValues)
        plt.ylabel("RMSE")
        plt.plot(a,testRMSEValues)
        plt.ylabel(" RMSE")
        plt.xlabel("max(p) ")
        fileName = "YachtPlot";
        plt.savefig(fileName)
