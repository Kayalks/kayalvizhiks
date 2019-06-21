
import numpy as np
import math as m
import random as rand
#import time
from numpy import inf
ion = np.recfromtxt("ionosphere.txt",delimiter = ",")
iris = np.recfromtxt("iris.txt",delimiter = ",")
def split_data(dataset):    
    split_size = 0.40
    train = []
    test = []
    rand.seed(0)
    data = list(dataset)
    for r in range(len(data)):
        if rand.random() > split_size:
            train.append(data[r])
        else:
            test.append(data[r])               
    return train,test 
def min_sort(t_list): 
    sort_list = list_sorting(t_list) 
    n_neigh=list()    
    for i in range(k):
            n_neigh.insert(i,sort_list[i])       
    return n_neigh

def list_sorting(t_list):
    s_list = list()
    tlen = len(t_list[0])-1
    while t_list:
        cur_min = t_list[0][tlen]
        n_tuple = t_list[0]
        for a in t_list:
            if cur_min>a[tlen]:
                cur_min = a[tlen]
                n_tuple = a
        s_list.append(n_tuple)
        t_list.remove(n_tuple)
    return s_list
def pred_e_distance(x_tr,x_t): # predict the euclidean distance and combine it with the sample set.
    dis = 0
    pred_set = list()
    test_len=len(x_t)-1
    for i in range(test_len):
        dis += pow(float(x_t[i])-float(x_tr[i]),2)
        pred_set.insert(i,float(x_tr[i]))
        
    pred_set.insert(i+1,x_tr[i+1])
    pred_set.insert(i+2, m.sqrt(dis))
    return   pred_set

def predict_NN(x_train,x_test): #identifies the nearest sample for each test sample
    train = []
    x_train = list(x_train)      
    #print(x_train)
    for j in range(len(x_train)):
        pred_set = pred_e_distance(x_train[j],x_test)
        train.append(pred_set)
    #print(train)
    near_neigh=[]
    near_neigh = min_sort(train)        
    #print(near_neigh)
    return near_neigh

def identify_NN(n_n): #identify the label of the nearest neighbour for K=n
    label_key={}
    for i in range(len(n_n)):
        l = n_n[i][-2]
        if l in label_key:
            label_key[l]+=1
        else:
            label_key[l]=1
    sort_label=[]
    if k>1:     
        while label_key:     
            currentvalue = next(iter(label_key.values()))
            n_tuple = next(iter(label_key.keys()))
            for key,val in label_key.items():
                if currentvalue<val:
                    currentvalue = val
                    n_tuple=key
            sort_label.append((n_tuple,currentvalue))
            del label_key[n_tuple]             
    else:
        sort_label.append((next(iter(label_key.keys())),next(iter(label_key.values()))))
    #print(label_key.values())
    return sort_label[0][0]

def predict_accuracy(X_test,y_predict): #find the accuracy and error rate for the predictions
    val= 0
    for i in range(0,len(X_test)-1):        
        if y_predict[i] == X_test[i][-1]:
            val+= 1
    accuracy= round(val/len(X_test),5)
    error_rate= round((1-accuracy),5)
    return accuracy,error_rate
def main_script(X):
    X_train=[]
    X_test=[]
    y_predict=[]
    X_train,X_test = split_data(X)
    #st=time.time()
    for i in range(0,len(X_test)-1):       
        all_NN = predict_NN(X_train,X_test[i])
        #print(all_NN)            
        pred_label = identify_NN(all_NN) 
        y_predict.append(pred_label)   
    #print(y_predict)
    #print(time.time()-st,"seconds")
    accuracy,error_rate = predict_accuracy(X_test,y_predict)  
    return accuracy,error_rate
#trigger:
k=1
accuracy,error_rate=main_script(iris)
print("\nIris Dataset for 1-NN:\nAccuracy: ",accuracy,"\nError Rate: ",error_rate)
k=3
accuracy,error_rate=main_script(iris)
print("\nIris Dataset for 3-NN:\nAccuracy: ",accuracy,"\nError Rate: ",error_rate)
k=1
accuracy,error_rate=main_script(ion)
print("\nIonosphere Dataset for 1-NN:\nAccuracy: ",accuracy,"\nError Rate: ",error_rate)
k=3
accuracy,error_rate=main_script(ion)
print("\nIonosphere Dataset for 3-NN:\nAccuracy: ",accuracy,"\nError Rate: ",error_rate)
