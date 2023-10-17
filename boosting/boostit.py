import numpy as np
from numpy import log as ln
from itertools import repeat
class BoostingClassifier:
    """ Boosting for binary classification.
    Please build an boosting model by yourself.

    Examples:
    The following example shows how your boosting classifier will be used for evaluation.
    >>> X_train, y_train = load_train_dataset() # we ignore the process of loading datset
    >>> X_test, y_test = load_test_dataset()
    >>> clf = BoostingClassifier().fit(X_train, y_train)
    >>> y_pred =  clf.predict(X_test) # this is how you get your predictions
    >>> evaluation_score(y_pred, y_test) # this is how we get your final score for the problem.

    """
    def __init__(self):
        # initialize the parameters here
        self.weight_list = []            #influence of the training points 
        self.confident_factor_list = []  #how good the model is
        self.model_T = []                #List of models   (plane, value of plane)
        self.prediction_list = []   
        self.epoch_T = 10         #number of desired model 

        pass
        
        
    def fit(self, X, y):
        """ Fit the boosting model.

        Parameters
        ----------
        X : { numpy.ndarray } of shape (n_samples, n_features)
            The input samples with dtype=np.float32.
        
        y : { numpy.ndarray } of shape (n_samples,)
            Target values. By default, the labels will be in {-1, +1}.

        Returns
        -------
        self : object
        """
        
        '''
                    Initial the list values 
        '''        
        [num, _] = X.shape
        self.weight_list.extend(repeat(1/len(X),len(X)))
        self.prediction_list = np.zeros((num, 1), dtype=np.int16)

        '''
                    Set up the centriods list
        '''
        length = len(X[0])
        centriod_1 = []
        centriod_Neg_1 = [] 
        
        for x in range(length):
            centriod_1.append(0)
            centriod_Neg_1.append(0)
        '''
                Begin Boosting algo 
        '''
        little_t = 1
        first_pass = False
        error_rate = 0
        part_2 = True
        while little_t <= self.epoch_T and error_rate < .5:
            '''
                Reset the new centriods values
            '''
            for x in range(len(centriod_1)):
                centriod_1[x] = 0
                centriod_Neg_1[x] = 0
            '''
                Calculate new centriods values
            '''
            neg_weight = pos_weight = 0
            for i in range(len(X)):
                if y[i] == 1:
                    pos_weight+=self.weight_list[i]
                    for j in range(len(X[i])):
                        centriod_1[j] = centriod_1[j] + (X[i][j]* self.weight_list[i])
                if y[i] == -1:
                    neg_weight+=self.weight_list[i]
                    for j in range(len(X[i])):
                        centriod_Neg_1[j] = centriod_Neg_1[j] + (X[i][j]* self.weight_list[i])
    
            for i in range(len(centriod_1)):
                centriod_1[i] = centriod_1[i] / pos_weight
                centriod_Neg_1[i] = centriod_Neg_1[i] / neg_weight
            ''' 
                Get boundary line 
            '''
            '''
                1. midpoint 
            ''' 
            midpoint_val = self.midpoint(centriod_1,centriod_Neg_1)
            '''
                2. line vector
            '''
            line_vec = self.lines(centriod_1,centriod_Neg_1)
            '''
                3. orthogonal plane
            '''
            decision_plane_val = self.plane(midpoint_val, line_vec)
            '''
                1st pass to add noise to the initial decision plane 
            '''
            
            if first_pass == False:
                decision_plane_val-=.2
                first_pass = True
            
            '''
                Calculate training prediction 
            '''
            for i in range(len(X)):
                self.prediction_list[i] = self.prediction_val(X[i],line_vec,decision_plane_val)
            '''
                Find Error Rate
            '''
            error_rate = self.error_rate_val(self.prediction_list, y)
            #print("current error is ", error_rate)
            if error_rate < .5:
                ''' 
                    Update Weights
                '''
                for i in range(len(self.prediction_list)):
                    #correct 
                    if self.prediction_list[i] == y[i]:
                        demon = 2 * (1 - error_rate)
                        new_w = self.weight_list[i] / demon
                        self.weight_list[i] = new_w
                    #incorrect
                    else:
                        new_w = self.weight_list[i] / (2*error_rate)
                        self.weight_list[i] = new_w
                '''
                    Save current model
                '''
                self.model_T.append([line_vec,decision_plane_val])
                self.confident_factor_list.append(self.confident_factor(error_rate))
                '''
                    PRINT INFO FOR PART 2
                '''
                if part_2 == False:
                    print("Iteration ", little_t, ":")
                    print("Error = ", error_rate)
                    print("Alpha = ", self.confident_factor(error_rate))
                    print("Factor to increase weights = ", (1/(2*error_rate)))
                    denum = 2 * (1 - error_rate)
                    print("Factor to decrease weights = ", (1/denum))
            else:
                continue
            little_t+=1
        return self

    def confident_factor(self,error):
        base = (1-error) / error
        return .5 * np.log(base)
    def error_rate_val(self,prediction, actual):
        total_error = 0
        for i in range(len(self.prediction_list)):
            if self.prediction_list[i] != actual[i]:
                total_error += 1
        return total_error/len(self.prediction_list)

    def prediction_val(self, input, line, plane_val):
        pred_val = np.dot(input,line)
        if pred_val > plane_val:
            pred_val = 1
        else:
            pred_val = -1
        return pred_val

    def midpoint(self,x,y):
        z = []
        for i in range(len(x)):
            newpt = (x[i] + y[i]) / 2
            z.append(newpt)  
        return z
    def lines(self,x,y):
        line = []
        for i in range(len(x)):
            newpt = (x[i] - y[i])
            line.append(newpt)
        return line
    def plane(self,point , line):
        equal = 0
        equal = np.dot(point, line)
        return equal
    def predict(self, X):
        """ Predict binary class for X.

        Parameters
        ----------
        X : { numpy.ndarray } of shape (n_samples, n_features)

        Returns
        -------
        y_pred : { numpy.ndarray } of shape (n_samples)
                 In this sample submission file, we generate all ones predictions.
        """
         
        y_pred = []
        for x in range(len(X)):
            sum_val = 0
            for m in range(len(self.model_T)):
                y = np.dot(self.model_T[m][0],X[x])
                if y > self.model_T[m][1]:
                    y = 1
                else:
                    y = -1
                y *= self.confident_factor_list[m]
                sum_val += y
            if sum_val > 0:
                y_pred.append(1)
            else:
                y_pred.append(-1)
        return y_pred



        #return np.ones(X.shape[0], dtype=int)

