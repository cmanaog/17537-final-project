import sys
import numpy as np
import random
import time
import pandas as pd
from sklearn.metrics import log_loss

class Ann(object):

    def __init__(self, n_in, n_classes, n_hidden, learning_rate, alpha):
        self.n_in = n_in
        self.n_classes = n_classes # number of classes in final output
        self.n_hidden = n_hidden
        self.n_out = n_hidden # shape of weights for final layer
        self.learning_rate = learning_rate
        self.weights = np.random.uniform(-0.05, 0.05, (self.n_in, self.n_hidden))
        self.output_weights = np.random.uniform(-0.05, 0.05, self.n_out) # output weights
        self.hidden_layer_out = np.zeros(self.n_out)
        self.first_layer_deriv = np.zeros(self.n_out)
        self.output_deriv = 0
        self.delta_k = 0
        self.delta_h = np.zeros((self.n_in, self.n_hidden))
        self.alpha = alpha

    def feed_forward(self, train_vec, store = False):
        '''
      @param: train_vec (1d list) - training examples
            [0] = M1 [0, 100]
            [1] = M2 [0, 100]
            [2] = P1 [0, 100]
            [3] = P2 [0, 100]
            [4] = F[0, 100]
        @return: result (array-like) - 1 x 2 array, probability of class 0 or class 1
        '''
        result = [None, None]
        data = np.matrix(train_vec) # 1 x n_in
        # print(np.append(np.array(data.dot(self.weights)), 1))
        # for each elem in data, make it 0, if it's neg

        first_layer = self.relu(data.dot(self.weights)) # 1 x n_hidden

        # the vector that feeds into the activation function
        first_layer_into_node = np.array(data.dot(self.weights)) # 1 x n_hidden 

        logit = first_layer
        result[1] = self.sigmoid(np.array(first_layer).dot(np.array(self.output_weights.transpose()))).flatten()[0] # probablity of class 1
        result[0] = 1 - result[1].flatten()[0] # probability of class 0
        # if result > 0.5: result = 1
        # else: result = 0

        if store: # save computation if we're not using backprop
            self.hidden_layer_out = first_layer
            self.output_deriv = self.sigmoid(np.array(first_layer) * np.array(self.output_weights), deriv = True)
            self.first_layer_deriv = self.relu(first_layer_into_node, deriv = True)
            #self.first_layer_deriv[-1] = 1 # We think this is bias
        return (logit, result)

    def update_weights(self, output, truth, train):
        '''
        @param: output (float) - output of neural net
        @param: truth (str) - supervised label of data, yes or no
        @param: train(1d list) - input values
        @return: None - updates weights of neural net using grad. desc.
        '''
        error = truth - output # constant La
        self.delta_k = error * self.output_deriv

        self.delta_h = np.array(self.first_layer_deriv) * np.array(self.output_weights) * np.array(self.delta_k)

        # update the weights b/w hidden layer and output
        gradient_top = self.learning_rate * np.array(self.delta_k) * np.array(self.hidden_layer_out)
        self.output_weights = self.output_weights + gradient_top 
        
        gradient_bottom = self.learning_rate * np.matrix(train).transpose() * np.matrix(self.delta_h)
        gradient_bottom = self.clip(gradient_bottom)
        self.weights = self.weights + gradient_bottom 


    def softmax(self, x, deriv = False):
        if deriv: # output_size x output_size
            res = np.zeros((x.shape[1], x.shape[1]))
            for i in range(res.shape[0]):
                for j in range(res.shape[1]):
                    if i == j: delt = 1
                    else: delt = 0
                    res[i,j] = x[0,i] * (delt - x[0,j])
            return res
        return np.exp(x)/sum(np.exp(x))

    def relu(self, x, deriv = False):
        if deriv:
            np.where(x > 1, 1, 0)
        return np.maximum(0, x)

    def sigmoid(self, net, deriv = False):
        '''
        @param: net (float) - a weighted sum
        @param: deriv (bool) - indicates whether to use derivative or not
        '''
        if deriv:
            return self.sigmoid(net) * (1 - self.sigmoid(net))
        else:
            return 1.0 / (1.0 + np.exp(-1.0 * net))

    def clip(self, grad_btm, gamma = (-1, 1)):
        a = np.maximum(gamma[0], grad_btm)
        b = np.minimum(gamma[1], a)
        return b


class Adv(Ann):

    def __init__(self, n_in, n_classes, n_hidden, learning_rate, alpha):
        super().__init__(n_in, n_classes, n_hidden, learning_rate, alpha)
        self.n_classes = n_classes
        self.alpha = alpha
        self.output_weights = np.random.uniform(-0.05, 0.05, (self.n_out, self.n_classes)) # output weights: n_hidden + 1 x n_classes


    def update_weights(self, output, truth, train, main_nn, main_train):
        # top: d ce / d soft * d soft / d relu * d relu / d main out * d main out / w
        # btm: d ce / d soft * d soft / d relu * d relu / d main out * d main out / main hid out * d main hid out / w
        
        # ADV UPDATES 

        error = -1 / self.output[int(truth)] # constant La
        self.delta_k = error * self.output_deriv

        self.delta_h = np.array(self.first_layer_deriv) * np.array(self.output_weights) * np.array(self.delta_k)

        # update the weights b/w hidden layer and output
        gradient_top = self.learning_rate * np.array(self.delta_k) * np.array(self.hidden_layer_out)
        gradient_bottom = self.learning_rate * np.matrix(train) * np.matrix(self.delta_h)

        # # MAIN NET UPDATES        

        grad_btm = np.matrix(main_train).transpose() * np.matrix(main_nn.first_layer_deriv) * np.matrix(self.weights) * np.matrix(self.delta_h)

        grad_btm = self.clip(grad_btm)
        print(grad_btm)
        if np.isnan(grad_btm).any():
         
         print(main_nn.weights)
        main_nn.weights = main_nn.weights - self.alpha * grad_btm
        # Adv weights update 
        gradient_top = self.clip(gradient_top)
        gradient_bottom = self.clip(gradient_bottom)

        self.weights = self.weights + gradient_bottom 
        self.output_weights = self.output_weights + gradient_top 


    def feed_forward(self, train_vec, store = False):
        '''
      @param: train_vec (1d list) - training examples
            [0] = M1 [0, 100]
            [1] = M2 [0, 100]
            [2] = P1 [0, 100]
            [3] = P2 [0, 100            [4] = F[0, 100]
        @return: result (array-like) - 1 x 2 array, probability of class 0 or class 1
        '''
        #result = [None] * self.n_classes
        data = np.matrix(train_vec) # 1 x n_in
        # print(np.append(np.array(data.dot(self.weights)), 1))
        # for each elem in data, make it 0, if it's neg

        first_layer = self.relu(data.dot(self.weights)) # 1 x n_hidden, c
        first_layer = np.array(first_layer) # adding bias term, 1 x n_hidden + 1

        # the vector that feeds into the activation function
        first_layer_into_node = np.array(data.dot(self.weights)) # 1 x n_hidden + 1
        result = self.softmax(first_layer.dot(self.output_weights).flatten()) # 1 x n_hidden + 1
        pred = np.argmax(result)

        if store: # save computation if we're not using backprop
            self.hidden_layer_out = first_layer
            self.output = result
            self.output_deriv = self.softmax(first_layer.dot(self.output_weights), deriv = True)
            self.first_layer_deriv = self.relu(first_layer_into_node, deriv = True)
            #self.first_layer_deriv[-1] = 1 # We think this is bias
        return result, pred 



def train(train, target, n_in, n_classes, n_hidden, adv_hidden, learning_rate = 0.1, alpha = 1):
    '''
    @param: train (2d list) - training input data
    @param: target (1d list) - output data from training data
    @param: n_in (int) - number of network inputs
    @param: n_hidden (int) - number of units in the hidden layer
    @param: learning_rate (int) - step for gradient descent, defaults to 0.05
    @return: network (Ann) - trained neural network
    '''
    # Create a feed-forward network with n_in inputs, n_hidden hidden units,
    # and n_out output units
    # Init all network weights to small random numbers (between [-0.05,0.05])
    n_classes_main, n_classes_adv = n_classes
    network = Ann(n_in, n_classes_main, n_hidden, learning_rate, alpha)
    adv = Adv(n_hidden, n_classes_adv, adv_hidden, learning_rate, alpha)
    count = 0

    pred_target, adv_target = target[0], target[1]
    while count < 1:
        error_rate = 0
        main_error = []
        adv_error = []
        # For each training vector and output,
        for row in range(train.shape[0]):
            # Input x to the network and computer output o_u
            print("MAIN FEED FORWARD")
            (logit, main_output) = network.feed_forward(train[row], store = True)
            main_error.append(main_output)

            # update weights of approval network (Ly)
            print("MAIN WEIGHT UPDATE")
            network.update_weights(np.argmax(main_output), pred_target[row], train[row])

            # get output for race
                # another feed_forward with logit as the input
            print("ADVERSARY FEED FORWARD")

            adv_softmax, adv_output = adv.feed_forward(logit, store = True)
            adv_error.append(adv_softmax)

            # # calculate the partial for main nn and update the weight for race nn (Ld)
            # # ld = network.update_weights(output_race, race_col_from_train_x, logit)
            print("ADVERSARY WEIGHT UPDATE")
            adv.update_weights(adv_output, adv_target[row], logit, network, train[row])

        

            # L = ly - alpha * Ld

            #if row % 100000 == 0: print(row)
        count += 1

        error_rate += log_loss(pred_target, main_error) - alpha * log_loss(adv_target, adv_error, labels = (0,1,2,3,4)) 
        print("cross entropy loss = " + str(error_rate))
    return network, adv

def main():
    '''
    @param: train (str) - path to training data (2d)
    @param: target (str) - path to target data (1d)
    @param: test (str) - path to testing data (2d)
    '''
    ################## TUNING ########################
    n_hidden = 5 # magic number, change to tune neural net
    adv_hidden = 5
    eta = 0.001 # magic number, change to tune neural net
    alpha = 1 # tuning param for adversary
    ##################################################
    print("Reading in data")
    train_data = pd.read_csv("train_clean.csv", nrows = 100) #nrows to read in less
    train_data = train_data.drop("Unnamed: 0", axis = 1)
    x_train = np.asarray(train_data.iloc[:, 1:-1], dtype = np.float64)
    y_train = (np.asarray(train_data["approved"], dtype = np.float64),
               np.asarray(train_data["applicant_race_name_1"], dtype = np.float64))
    print("Completed reading in data")

    test_data = pd.read_csv("test_clean.csv", nrows = 100)
    test_data = test_data.drop("Unnamed: 0", axis = 1)
    x_test = np.asarray(test_data.iloc[:, 1:-1], dtype = np.float64)
    y_test = (np.asarray(test_data["approved"], dtype = np.float64),
              np.asarray(test_data["applicant_race_name_1"], dtype = np.float64))

    n_in = x_train.shape[1]
    n_classes = 2, 5#len(set(y_test[0])), len(set(y_test[1]))
    start = time.time()
    trained_nn, trained_adv = train(x_train, y_train, n_in, n_classes, n_hidden, adv_hidden, eta, alpha)
    end = time.time()
    sys.stdout.write("TRAINING COMPLETED! NOW PREDICTING.\n")
    #for row in x_test:
        #print(row)
    #    output = trained_nn.feed_forward(row)
    #    sys.stdout.write(str(np.round(output[1])) + "\n")

main()
