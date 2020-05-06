#made by sonhaksun

import numpy as np
import random

Max_Update = 100
Epsilon = 0.0001
Initial_Learning_Rate = 0.3

def train_LR(d, label):
    '''
    :param data: data without bias
    :param label: label
    :return:
    '''
    Learning_Rate = Initial_Learning_Rate
    ex_size = len(d)
    dim_size = len(d[0])
    theta = np.array([[random.random() for i in range(dim_size+1)]])
    theta = theta.T
    data = np.ones((ex_size,dim_size+1))
    normalize = np.zeros((dim_size,2))
    mean_error_rate = []

    #copy data
    for i in range(ex_size):
        for j in range(dim_size):
            data[i][j+1] = d[i][j]

    #normalize data, set data 0 to 1
    for j in range(dim_size):
        max_val = -2100000000
        min_val = 2100000000
        for i in range(ex_size):
            x = data[i][j+1]
            if x>max_val:
                max_val = x
            if x<min_val:
                min_val = x
        interval = max_val - min_val
        normalize[j][0] = min_val
        normalize[j][1] = interval

        for i in range(ex_size):
            data[i][j+1] = (data[i][j+1] - min_val) / interval


    #print(theta)
    i=0
    while(i<=Max_Update):
        diff = data @ theta - label
        square = diff.T @ diff
        rms = square[0][0] / ex_size

        if i>=1:
            if mean_error_rate[-1] - rms < 0: #learning rate is too large that error rate increases
                Learning_Rate /= 3.16 #multiple root(10)
                i=0
                print('decreased learning rate to ', Learning_Rate)
            elif (mean_error_rate[-1] - rms) < Epsilon:
                mean_error_rate.append(rms)
                break
            if i==Max_Update: #learning rate is too small
                Learning_Rate *= 3.16
                i=0
                print('increased learning rate to ', Learning_Rate)



        mean_error_rate.append(rms)
        change_theta = Learning_Rate/ex_size*(data.T@diff)
        theta = theta - change_theta
        #print(theta)
        i+=1

    print(theta)
    print(mean_error_rate[-1])
    return theta, normalize

def test_LR(X,Y,theta,normalize):
    ex_size = len(X)
    dim_size = len(X[0])
    #set bias and copy data
    data = np.ones((ex_size, dim_size + 1))
    for i in range(ex_size):
        for j in range(dim_size):
            data[i][j + 1] = X[i][j]

    #normalizing
    for j in range(dim_size):
        min_val = normalize[j][0]
        interval = normalize[j][1]
        for i in range(ex_size):
            data[i][j+1] = (data[i][j+1]-min_val)/interval
    #test
    diff = data @ theta - Y
    square = diff.T @ diff
    rms = square[0][0] / ex_size
    print(rms)

    return rms



train_X = np.array([[1,1,1],[2,2,2],[3,3,2],[4,4,5],[4,3,3],[1,2,1],[-1,-1,-1],[-2,-1,-2]])
train_Y = np.array([[2,3,4,5,4,2,0,-1]])
train_Y = train_Y.T

test_X = np.array([[0,0,0], [-1.2,-1.2,-1.5], [-3,-3,-3]])
test_Y = np.array([[1,-0.3,-2]])
test_Y = test_Y.T

theta, normalize = train_LR(train_X, train_Y)
test_LR(test_X,test_Y,theta,normalize)