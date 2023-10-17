import math
import csv
import numpy as np
from numpy import genfromtxt




def run(Xtrain_file, Ytrain_file, test_data_file, pred_file):
	#Read in the x training data 
	input_x_train_file_list = genfromtxt(Xtrain_file, delimiter = ',')

	#Read in the training label data 
	y_train_file_list = genfromtxt(Ytrain_file, delimiter = ' , ')

	#Create the test list 
	test_data_file_list = genfromtxt(test_data_file,delimiter = ',') 
	#Create the prediction list
	num = len(test_data_file_list)
	prediction = np.zeros((num, 1), dtype=np.int16)

	#Set K value
	K = 13
	#Predict the test case values 
	for x in range(len(test_data_file_list)):
		#create list of shortest distance (smallest ----> largest)
		list_of_dist = []
		list_of_dist = listOfDist(input_x_train_file_list, test_data_file_list[x], y_train_file_list)
		class_label_vote = [0,0,0,0,0,0,0,0,0,0,0]
		#predict value based on 1st K values 
		for k in range(K):
			class_label_vote[int(list_of_dist[k][1])] += 1 
		prediction_value = retrieveLabel(class_label_vote)
		prediction[x] = prediction_value
	
	np.savetxt(pred_file, prediction, fmt='%1d', delimiter=",")



def retrieveLabel(x):
	highest_class_value = x[0]
	highest_class_label = 0

	for i in range(1,len(x)):
		if x[i] > highest_class_value:
			highest_class_label = i
			highest_class_value = x[i]
	return highest_class_label

def listOfDist(input_x_train_file_list, test_point_position, y_train_file_list):
	#Find the euclidian dist on all values
	new_euclidian = 0
	list_of_dist = []
	for x in range(len(input_x_train_file_list)): 
		new_euclidian = euclidian(test_point_position, input_x_train_file_list[x])
		new_val = [new_euclidian, y_train_file_list[x]]
		placed = False 
		position = 0
		if len(list_of_dist) == 0:
			list_of_dist.append(new_val)
		else:
			while not placed and position < len(list_of_dist):
				if new_euclidian < list_of_dist[position][0]:
					list_of_dist.insert(position  , new_val)
					placed = True
				elif new_euclidian == list_of_dist[position][0]:
					if y_train_file_list[x] < list_of_dist[position][1]:
						list_of_dist.insert(position -1, new_val)
					else:
						position+=1
				else:
					position+=1
			if placed == False:
				list_of_dist.append(new_val) #making a dupliacate 
	return list_of_dist

def euclidian(x, y):
    dist = 0
    for i in range(len(x)):
        val = x[i] - y[i]
        val**2
        val = abs(val)
        val = math.sqrt(val)
        dist+= val
    return dist


if __name__ == "__main__":
    Xtrain_file = 'Xtrain.csv'
    Ytrain_file = 'Ytrain.csv'
    test_data_file = 'testing.txt'
    pred_file = 'result'
    run(Xtrain_file,Ytrain_file,test_data_file,pred_file)