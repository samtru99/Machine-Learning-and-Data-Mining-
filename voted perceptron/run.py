import numpy as np 
from numpy import genfromtxt
#import math
import csv 


def run(Xtrain_file, Ytrain_file, test_data_file, pred_file):
	
    #Read in YTrain_file
    y_train_file_list = []
    with open(Ytrain_file, 'r') as output:
        for x in output:
            y_train_file_list.append(int(x))
    #correct the 0's to -1's
    for x in range(len(y_train_file_list)):
        if y_train_file_list[x] == 0:
            y_train_file_list[x] = -1
      
    #Read in Xtrain_file, check if this is correct
    input_x_train_file_list = genfromtxt(Xtrain_file, delimiter = ',') #works so far

    #Create the test_data_file_list
    test_data_file_list = genfromtxt(test_data_file,delimiter = ',') #for when to submit 
    
   
    #Create the pred_file 
    num = len(test_data_file_list) 

    prediction = np.zeros((num, 1), dtype=np.int16)
    

    #VOTE PERCEPTION EQ

    #initialize all values 
    k = 0
    c_k = []      #list of weights votes
    c_k.append(0)
    w_k = []      #list of list of weights 

    init = np.zeros((1,len(input_x_train_file_list[0])), dtype=np.int16) #need an empty array
    w_k.append(init)
    little_t = 0 
    bit_T = 1#hyper parameter
    
    #Run vote perception 
    while little_t < bit_T:
        for x in range(int(len(input_x_train_file_list))): 
            value = np.dot(w_k[k], input_x_train_file_list[x])
            if y_train_file_list[x] * value  <= 0:
                new_w = [] 
                new_w = w_k[k] + np.dot(y_train_file_list[x],input_x_train_file_list[x])
                w_k.append(new_w)
                k += 1
                c_k.append(1)
            else:
                c_k[k] += 1
        little_t += 1
    
    #Run on testing data 
    for x in range(len(test_data_file_list)):
        little_k = 1
        average_sum = 0
        while little_k < len(w_k):
            dot_sum = np.dot(w_k[little_k], test_data_file_list[x])
            if dot_sum <= 0:
                dot_sum = -1 
            else:
                dot_sum = 1
            average_sum += c_k[little_k] * dot_sum
            little_k+=1

        if average_sum > 0:
            average_sum = 1
        else:
            average_sum = 0 #orginial was average_sum which was -1
        prediction[x] = average_sum  


    #Convert the prediction to 1's and 0's
    for x in range(len(prediction)):
        if prediction[x] == -1:
            prediction[x] = 0

    #Convert the yTrain label back to original 
    for x in range(len(y_train_file_list)):
        if y_train_file_list[x] == -1:
            y_train_file_list[x] = 0


    '''correct = 0
    for x in range(len(prediction)):
        if prediction[x] == y_train_file_list[450+x]:
            correct += 1
    '''
    np.savetxt(pred_file, prediction, fmt='%1d', delimiter=",")
    
    
if __name__ == "__main__":
    Xtrain_file = 'Xtrain.csv'
    Ytrain_file = 'Ytrain.csv'
    test_data_file = 'testing2.txt'
    pred_file = 'result'
    run(Xtrain_file,Ytrain_file,test_data_file,pred_file)