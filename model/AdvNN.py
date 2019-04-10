import sys
import numpy as np 
import random
import time

class Ann(object):
    
    def __init__(self, n_in, n_out, n_hidden, learning_rate):
        self.n_in = n_in
        self.n_out = n_out
        self.n_hidden = n_hidden
        self.learning_rate = learning_rate
        self.weights = np.random.uniform(-0.05, 0.05, (n_in, n_hidden))
        self.out = np.random.uniform(-0.05, 0.05, n_out) # output weights
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
        data = np.matrix(train_vec)
        # print(np.append(np.array(data.dot(self.weights)), 1))
        first_layer = sigmoid(data.dot(self.weights)) # 1 x 3 (# hidden units)
        first_layer = np.append(np.array(first_layer), 1) # bias term, 1 x 4
        first_layer_deriv = np.append(np.array(data.dot(self.weights)), 1)
        result = sigmoid(first_layer.dot(self.out)) # should be constant (1 x 1)
        if store: # save computation if we're not using backprop
            self.hidden_layer_out = first_layer
            self.output_deriv = sigmoid(first_layer.dot(self.out), True)
            self.first_layer_deriv = sigmoid(first_layer_deriv, True)
            self.first_layer_deriv[-1] = 1
        return result * 100

    def update_weights(self, output, truth, train):
        '''
        @param: output (float) - output of neural net
        @param: truth (str) - supervised label of data, yes or no
        @param: train(1d list) - input values
        @return: None - updates weights of neural net using grad. desc.
        '''
        error = truth - output # constant
        self.delta_k = error * self.output_deriv
        self.delta_h = self.first_layer_deriv * self.out * self.delta_k 
        gradient_top = self.learning_rate * self.delta_k * self.hidden_layer_out
        self.out = self.out + gradient_top
        self.delta_h = np.delete(self.delta_h, -1) # remove bias
        gradient_bottom = self.learning_rate * np.matrix(train).transpose() * np.matrix(self.delta_h)  
        self.weights = self.weights + gradient_bottom

def norm(arr):
    '''
    @param: arr (1d array) - array to normalize
    @return: new_arr (1d array) - normalized array
    '''
    return arr / 100

def parse(data):
    '''
    @param: data (2d list) - unnormalized data 
    @return: new (2d list) - nornalized data and biased added
    '''
    new = data
    return np.apply_along_axis(norm, 1, data)

def sigmoid(net, deriv = False):
    '''
    @param: net (float) - a weighted sum
    @param: deriv (bool) - indicates whether to use derivative or not
    '''
    if deriv:
        return sigmoid(net) * (1 - sigmoid(net))
    else:
        return 1.0 / (1.0 + np.exp(-1.0 * net))

def backpropogate(train, target, n_in, n_out, n_hidden, learning_rate = 0.1):
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
            output = network.feed_forward(train[row], store = True)
            # update weights
            network.update_weights(output, target[row], train[row])
            error_rate += (target[row] - output)**2
        count += 1
        sys.stdout.write(str(0.5 * error_rate) + "\n")
    return network

def main(train, target, test):
    '''
    @param: train (str) - path to training data (2d)
    @param: target (str) - path to target data (1d)
    @param: test (str) - path to testing data (2d)
    '''
    ################## TUNING ########################
    n_hidden = 4 # magic number, change to tune neural net
    eta = 0.01 # magic number, change to tune neural net
    ##################################################
    train_data = np.genfromtxt(train, delimiter = ",", dtype = float, skip_header = 1)
    train_data = parse(train_data)
    keys = np.genfromtxt(target, dtype = float) # target data
    test_data = np.genfromtxt(test, delimiter = ",", dtype = float, skip_header = 1)
    test_data = parse(test_data)
    n_in = train_data.shape[1]
    n_out = n_hidden + 1# only one output
    start = time.time()
    trained_nn = backpropogate(train_data, keys, n_in, n_out, n_hidden, eta)
    end = time.time()
    sys.stdout.write("TRAINING COMPLETED! NOW PREDICTING.\n")
    for row in test_data:
        output = trained_nn.feed_forward(row)
        sys.stdout.write(str(np.round(output)) + "\n")

if __name__ == "__main__":
    # file1, file2, file3 = (sys.argv[1], sys.argv[2], sys.argv[3])
    file1, file2, file3 = ("education_train.csv",
                          "education_train_keys.txt",
                          "education_dev.csv")
    main(file1, file2, file3)

