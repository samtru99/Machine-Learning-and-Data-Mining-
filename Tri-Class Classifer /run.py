import numpy as np
import math
'''
train_input_dir : the directory of training dataset txt file. For example 'training1.txt'.
train_label_dir :  the directory of training dataset label txt file. For example 'training1_label.txt'
test_input_dir : the directory of testing dataset label txt file. For example 'testing1.txt'
pred_file : output directory 
'''

def run (train_input_dir,train_label_dir,test_input_dir,pred_file):
    #Reading data
    trainingData = np.loadtxt(train_input_dir, skiprows = 0)
    test_data = np.loadtxt(test_input_dir,skiprows=0) 
    trainingLabel = np.loadtxt(train_label_dir, skiprows = 0)
    #test_output = np.loadtxt("testing2_label.txt", skiprows = 0)
    [num, _] = test_data.shape
    prediction = np.zeros((num, 1), dtype=np.int16)
    #Insert the Training Data 
    length = 0
    length = len(trainingData[0])
    centriod0 = []
    centriod1 = []
    centriod2 = []
    for x in range(length):
        centriod0.append(0)
        centriod1.append(0)
        centriod2.append(0)

    #Insert the Training Label 
    
    #Sum up classes vector values 
    total0 = total1 = total2 = 0
    for x in range(len(trainingLabel)):
         if trainingLabel[x] == 0:
            total0 = total0 + 1 
            for y in range(len(trainingData[x])):
                centriod0[y] = centriod0[y] + trainingData[x][y]   
         if trainingLabel[x] == 1:
            total1 = total1 + 1 
            for y in range(len(trainingData[x])):
                centriod1[y] = centriod1[y] + trainingData[x][y]
         if trainingLabel[x] == 2:
            total2 = total2 + 1 
            for y in range(len(trainingData[x])):
                centriod2[y] = centriod2[y] + trainingData[x][y]

    #Avg out vectors for the centriod value
    for x in range(len(centriod0)):   
        centriod0[x] = centriod0[x] / total0
        centriod1[x] = centriod1[x] / total1
        centriod2[x] = centriod2[x] / total2

    #Calculate the midpoint
    
    mp01 = midpoint(centriod0,centriod1)
    mp02 = midpoint(centriod0,centriod2)
    mp12 = midpoint(centriod1,centriod2)
    
    
    #Calculate the line
    line01 = []
    line02 = []
    line12 = []
    line01 = lines(centriod0,centriod1)
    line02 = lines(centriod0,centriod2)
    line12 = lines(centriod1,centriod2)

   

    #calculate the plane
    plane01Val = plane(mp01,line01)
    plane02Val = plane(mp02,line02)
    plane12Val = plane(mp12,line12)
 
    #Predict the values
    for x in range(len(test_data)):
        predictVal = np.dot(test_data[x], line01)
        if predictVal >= plane01Val:
            #check if it's 0 or 2
            predictVal = np.dot(test_data[x], line02)
            if predictVal >= plane02Val:
                #its 0
                prediction[x] = 0
            else:
                prediction[x] = 2
        #Check if it's 1 or 2
        else:
            predictVal = np.dot(test_data[x], line12)
            #its 1
            if predictVal >= plane12Val:
                prediction[x] = 1
            #its 0
            else:
                prediction[x] = 2
    
    #print(prediction)

    # Saving you prediction to pred_file directory (Saving can't be changed)
    np.savetxt(pred_file, prediction, fmt='%1d', delimiter=",")

    '''correct = 0
    for x in range(len(prediction)):
        #correct = 0
        if prediction[x] == test_output[x]:
            correct+=1
    print("total correct ", correct)
    '''
def euclidian(x, y):
    dist = 0
    for i in range(len(x)):
        val = x[i] - y[i]
        val**2
        val = abs(val)
        val = math.sqrt(val)
        dist+= val
def midpoint(x,y):
    z = []
    for i in range(len(x)):
        newpt = (x[i] + y[i]) / 2
        z.append(newpt)
    return z
def lines(x,y):
    line = []
    for i in range(len(x)):
        newpt = (x[i] - y[i])
        line.append(newpt)
    return line
def plane(point , line):
    equal = 0
    equal = np.dot(point, line)
    return equal

    
if __name__ == "__main__":
    train_input_dir = 'training1.txt'
    train_label_dir = 'training1_label.txt'
    test_input_dir = 'testing1.txt'
    pred_file = 'result'
    run(train_input_dir,train_label_dir,test_input_dir,pred_file)
