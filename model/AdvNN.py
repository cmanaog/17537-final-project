import sys
import numpy as np
import random
import time
import pandas as pd

class Ann(object):

    def __init__(self, n_in, n_out, n_hidden, learning_rate):
        self.n_in = n_in
        self.n_out = n_out
        self.n_hidden = n_hidden
        self.learning_rate = learning_rate
        self.weights = np.random.uniform(-0.05, 0.05, (n_in, n_hidden))
        self.out = np.random.uniform(-0.05, 0.05, n_out) # output weights: 1 x n_hidden
        self.hidden_layer_out = np.zeros(n_out)
        self.first_layer_deriv = np.zeros(n_out)
        self.output_deriv = 0
        self.delta_k = 0
        self.delta_h = np.zeros((n_in, n_hidden))

    def feed_forward(self, train_vec, store = False):
        '''
      @param: train_vec (1d list) - training examples
            [0] = M1 [0, 100]
            [1] = M2 [0, 100]
            [2] = P1 [0, 100]
            [3] = P2 [0, 100]
            [4] = F[0, 100]
        @return: result (int) - 0 or 1, classification of node
        '''
        data = np.matrix(train_vec) # 1 x n_in
        # print(np.append(np.array(data.dot(self.weights)), 1))
        # for each elem in data, make it 0, if it's neg
        #
        first_layer = self.relu(data.dot(self.weights)) # 1 x n_hidden
        first_layer = np.append(np.array(first_layer), 1) # adding bias term, 1 x n_hidden + 1

        # the vector that feeds into the activation function
        first_layer_into_node = np.append(np.array(data.dot(self.weights)), 1) # 1 x n_hidden + 1

        # TODO: logit = apply sigmoid to each elem of first_layer
        logit = first_layer

        result = self.sigmoid(first_layer.dot(self.out)) # probablity of 0 or 1
        if result > 0.5: result = 1
        else: result = 0

        if store: # save computation if we're not using backprop
            self.hidden_layer_out = first_layer
            self.output_deriv = self.sigmoid(first_layer.dot(self.out), True)
            self.first_layer_deriv = map(self.relu_deriv, first_layer_into_node)
            self.first_layer_deriv[-1] = 1 # We think this is bias
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
        self.delta_h = self.first_layer_deriv * self.out * self.delta_k

        # update the weights b/w hidden layer and output
        gradient_top = self.learning_rate * self.delta_k * self.hidden_layer_out
        self.out = self.out + gradient_top


        # update the weights b/w hidden layer and input
        self.delta_h = np.delete(self.delta_h, -1) # remove bias
        gradient_bottom = self.learning_rate * np.matrix(train).transpose() * np.matrix(self.delta_h)
        self.weights = self.weights + gradient_bottom


    def softmax(x):
        return np.exp(x)/sum(np.exp(x))

    def relu(x):
        return np.maximum(np.zeros(self.n_hidden), x)

    def relu_deriv(x):
        if (x > 0): return 1
        else: return 0

    def sigmoid(net, deriv = False):
        '''
        @param: net (float) - a weighted sum
        @param: deriv (bool) - indicates whether to use derivative or not
        '''
        if deriv:
            return sigmoid(net) * (1 - sigmoid(net))
        else:
            return 1.0 / (1.0 + np.exp(-1.0 * net))

def train(train, target, n_in, n_out, n_hidden, learning_rate = 0.1):
    '''
    @param: train (2d list) - training input data
    @param: target (1d list) - output data from training data
    @param: n_in (int) - number of network inputs
    @param: n_hidden (int) - number of units in the hidden layer
    @param: learning_rate (int) - step for gradient descent, defaults to 0.05
    @return: network (Ann) - trained neural network
    '''
    # Create a feed-forward network with n_in inpputs, n_hidden hidden units,
    # and n_out output units
    # Init all network weights to small random numbers (between [-0.05,0.05])

    network = Ann(n_in, n_out, n_hidden, learning_rate)
    count = 0
    while count < 1250:
        error_rate = 0
        # For each training vector and output,
        for row in range(train.shape[0]):
            # Input x to the network and computer output o_u
            (logit, output) = network.feed_forward(train[row], store = True)

            # update weights of approval network (Ly)
            la = network.update_weights(output, target[row], train[row])

            # get output for race
                # another feed_forward with logit as the input

            # update the weight for race nn (Ld)
            # ld = network.update_weights(output_race, race_col_from_train_x, logit)

            # L = ly - alpha * Ld


            error_rate += (target[row] - output)**2
        count += 1
        print("error rate = " + str(0.5 * error_rate))
    return network

def main():
    '''
    @param: train (str) - path to training data (2d)
    @param: target (str) - path to target data (1d)
    @param: test (str) - path to testing data (2d)
    '''
    ################## TUNING ########################
    n_hidden = 256 # magic number, change to tune neural net
    eta = 0.01 # magic number, change to tune neural net
    ##################################################
    train_data = pd.read_csv("~/Downloads/train_clean.csv")
    x_train = np.asarray(train_data.iloc[:, 1:-1], dtype = np.float64)
    y_train = np.asarray(train_data.iloc[:, -1], dtype = np.float64)

    test_data = pd.read_csv("~/Downloads/test_clean.csv")
    x_test = np.asarray(test_data.iloc[:, 1:-1], dtype = np.float64)
    y_test = np.asarray(test_data.iloc[:, -1], dtype = np.float64)

    n_in = x_train.shape[1]
    n_out = n_hidden + 1 # only one output
    start = time.time()
    trained_nn = train(x_train, y_train, n_in, n_out, n_hidden, eta)
    end = time.time()
    sys.stdout.write("TRAINING COMPLETED! NOW PREDICTING.\n")
    for row in test_data:
        output = trained_nn.feed_forward(row)
        sys.stdout.write(str(np.round(output)) + "\n")

main()
