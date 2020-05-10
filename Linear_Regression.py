#made by sonhaksun

import numpy as np
import random

Max_Update = 1000
Epsilon = 0.001
Initial_Learning_Rate = 0.3

def train_LR(d, label, lmda=0, random_seed = None):
    '''

    :param data: data without bias
    :param label: label
    :param lmda: regularization coef.
    :return:
    '''
    Learning_Rate = Initial_Learning_Rate
    ex_size = len(d)
    dim_size = len(d[0])
    if random_seed: random.seed(random_seed)
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
        if interval==0: interval=1
        normalize[j][0] = min_val
        normalize[j][1] = interval

        for i in range(ex_size):
            data[i][j+1] = (data[i][j+1] - min_val) / interval


    #print(theta)
    i=0
    while(i<=Max_Update):
        diff = data @ theta - label
        square = diff.T @ diff
        mer = square[0][0] / ex_size

        if i>=1:
            if mean_error_rate[-1] - mer < 0: #learning rate is too large that error rate increases
                Learning_Rate /= 3.16 #multiple root(10)
                i=0
                print('decreased learning rate to ', Learning_Rate)
            elif (mean_error_rate[-1] - mer) < Epsilon:
                mean_error_rate.append(mer)
                break
            if i==Max_Update: #learning rate is too small
                Learning_Rate *= 3.16
                i=0
                print('increased learning rate to ', Learning_Rate)
                break



        mean_error_rate.append(mer)
        change_theta = Learning_Rate/ex_size*(data.T@diff) + lmda*theta
        theta = theta - change_theta
        #print(theta)
        i+=1

    #print(theta)
    #print(mean_error_rate[-1])
    accuracy = mean_error_rate[-1]
    return theta, normalize, accuracy

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
    mer = square[0][0] / ex_size
    #print(mer)

    return mer

def k_fold(k,data,label,lmda=0.0, random_seed = None):
    def concat_all(data):  # concatinate all array in a list, all elements should have same number of columns
        dat = data[0]
        for ind in range(1, len(data)):
            dat = np.concatenate((dat, data[ind]), axis=0)
        return dat



    folds_X = []
    folds_Y = []
    ex_size = len(data)
    acc_train = []
    acc_test = []

    #shuffle data
    if random_seed:
        np.random.seed(random_seed)
    idx = np.arange(ex_size)
    np.random.shuffle(idx)
    X = data[idx]
    y = label[idx]

    #divide folds
    for i in range(k):
        start_idx = i*ex_size//k
        end_idx = (i+1)*ex_size//k
        folds_X.append(X[start_idx:end_idx])
        folds_Y.append(y[start_idx:end_idx])

    for i in range(k):
        X_train = folds_X[:]
        X_test = X_train.pop(i)
        y_train = folds_Y[:]
        y_test = y_train.pop(i)
        X_train = concat_all(X_train)
        y_train = concat_all(y_train)

        theta,normalize,accuracy_train = train_LR(X_train,y_train,lmda=lmda,random_seed=random_seed+3)
        accuracy_test = test_LR(X_test,y_test,theta,normalize)
        acc_train.append(accuracy_train)
        acc_test.append(accuracy_test)

    #print(acc_train)
    #print(acc_test)


    print("training accuracy", np.mean(acc_train))
    print("test accuracy", np.mean(acc_test))

    return np.mean(acc_train), np.mean(acc_test)

def best_lambda(k,data,label):
    seed = 41
    print("\nlambda: 0.3")
    train,test = k_fold(k,data,label,lmda=0.3,random_seed=seed)
    best_l = 0.3
    best_acc = test
    mul = 1/1.2
    cur_l = best_l*mul


    cnt = 0
    while(cnt<=20):
        cur_l *= mul
        print("\nlambda: ", cur_l)
        x,cur_acc = k_fold(k,data,label,lmda=cur_l,random_seed=seed)
        if cur_acc<best_acc:
            best_acc=cur_acc
            best_l = cur_l

        cnt+=1


    print("\nbest accuracy: ", best_acc)
    print("best lambda: ",best_l)
    return best_acc,best_l

train_X = np.array([[1,1,1],[2,2,2],[3,3,2],[4,4,5],[4,3,3],[1,2,1],[-1,-1,-1],[-2,-1,-2]])
train_Y = np.array([[2,3,4,5,4,2,0,-1]])
train_Y = train_Y.T

test_X = np.array([[0,0,0], [-1.2,-1.2,-1.5], [-3,-3,-3]])
test_Y = np.array([[1,-0.3,-2]])
test_Y = test_Y.T

kf_X = np.array([[1,1,1],[2,2,2],[3,3,2],[4,4,5],[4,3,3],[1,2,1],[-1,-1,-1],[-2,-1,-2], [0,0,0], [-1.2,-1.2,-1.5], [-3,-3,-3]])
kf_Y = np.array([[2,3,4,5,4,2,0,-1,1,-0.3,-2]])
kf_Y = kf_Y.T

#k_fold(3,kf_X,kf_Y)
#theta, normalize = train_LR(train_X, train_Y)
#test_LR(test_X,test_Y,theta,normalize)
