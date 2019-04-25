# -*- coding: utf-8 -*-
"""Adv_final.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1jSDiigmtYDBHYwneYq9W24fLSjpmseFu
"""
import sys
import numpy as np
import random
import time
import pandas as pd
import pickle
from sklearn.metrics import log_loss


class Ann(object):

    def __init__(self, n_in, n_classes, n_hidden, learning_rate, alpha):
        self.n_in = n_in
        self.n_classes = n_classes # number of classes in final output
        self.n_hidden = n_hidden
        self.n_out = n_hidden # shape of weights for final layer
        self.learning_rate = learning_rate
        self.weights = np.random.uniform(-0.05, 0.05, (self.n_in, self.n_hidden))
        self.output_weights = np.random.uniform(-0.05, 0.05, (self.n_out, self.n_classes)) # output weights
        self.hidden_layer_out = np.zeros(self.n_out)
        self.first_layer_deriv = np.zeros(self.n_out)
        self.output_deriv = 0
        self.delta_k = 0
        self.delta_h = np.zeros((self.n_in, self.n_hidden))
        self.alpha = alpha

    def feed_forward(self, train_vec, target, store = False):
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
        res = np.matrix(first_layer) * (np.matrix(self.output_weights))
        res = np.asarray(res).flatten()
        result = self.softmax(res, target).flatten() # probablity of class 1
        #result[0] = 1 - result[1].flatten()[0] # probability of class 0
        # if result > 0.5: result = 1
        # else: result = 0

        if store: # save computation if we're not using backprop
            self.hidden_layer_out = first_layer
            self.output_deriv =  self.softmax(res, target, deriv = True) #self.sigmoid(np.array(first_layer) * np.array(self.output_weights), deriv = True)
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
        #error =  # constant La
        self.delta_k = self.output_deriv

        self.delta_h = np.matrix(self.first_layer_deriv) * np.matrix(self.output_weights) * np.matrix(self.delta_k).transpose()

        # update the weights b/w hidden layer and output
        gradient_top = self.learning_rate * np.matrix(self.hidden_layer_out).transpose() * np.matrix(self.delta_k)
        self.output_weights = self.output_weights + gradient_top 
        
        gradient_bottom = self.learning_rate * np.matrix(train).transpose() * np.matrix(self.delta_h)
        gradient_bottom = self.clip(gradient_bottom)
        self.weights = self.weights + gradient_bottom 


    def softmax(self, x, y = None, deriv = False):
        x_norm = (x - np.min(x)) / (np.max(x) - np.min(x) + 1)
        probs = np.exp(x_norm) / sum(np.exp(x_norm))
        if deriv: #d cross / d soft * d soft / d relu output, no need for dJ in update_weights
          probs = probs.flatten()
          probs[int(y)] = probs[int(y)] - 1
        return probs

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

        # print(self.output)
        # print(truth)
        # print(self.output[int(truth)])
        #error = -1 / self.output[int(truth)] # constant La, error is now included in self.output_deriv
        self.delta_k = np.matrix(self.output_deriv) # 1 x 5
        self.delta_h = np.matrix(self.first_layer_deriv).transpose() * (np.matrix(self.output_weights) * self.delta_k.transpose()).transpose()

        # update the weights b/w hidden layer and output
        gradient_top = self.learning_rate * np.matrix(self.hidden_layer_out).transpose() * np.matrix(self.delta_k)
        gradient_bottom = self.learning_rate * np.matrix(train) * np.matrix(self.delta_h)

        # # MAIN NET UPDATES 
        grad_btm = np.matrix(main_train).transpose() * np.matrix(main_nn.first_layer_deriv) #* np.matrix(self.weights) * np.matrix(self.delta_h)
        grad_btm = self.clip(grad_btm)

        if np.isnan(grad_btm).any():
            #print(grad_btm)
            #print(main_nn.weights)
            raise Exception("Becomes NAN")
        main_nn.weights = main_nn.weights - self.alpha * grad_btm
        # Adv weights update 
        gradient_top = self.clip(gradient_top)
        gradient_bottom = self.clip(gradient_bottom)

        self.weights = self.weights + gradient_bottom 
        self.output_weights = self.output_weights + gradient_top 


    def feed_forward(self, train_vec, target, store = False):
        '''
      @param: train_vec (1d list) - training examples
            [0] = M1 [0, 100]
            [1] = M2 [0, 100]
            [2] = P1 [0, 100]
            [3] = P2 [0, 100            [4] = F[0, 100]
        @return: result (array-like) - 1 x 2 array, probability of class 0 or class 1
        '''
     
        data = np.matrix(train_vec) # 1 x n_in

        first_layer = self.relu(data.dot(self.weights)) # 1 x n_hidden, c
        first_layer = np.array(first_layer) # adding bias term, 1 x n_hidden + 1
        if not np.any(first_layer):
            raise Exception("weights ", self.weights)


        # the vector that feeds into the activation function
        first_layer_into_node = np.array(data.dot(self.weights)) # 1 x n_hidden + 1
        x = np.asarray(first_layer.dot(self.output_weights))
        result = self.softmax(x.flatten()) # 1 x n_hidden + 1

        pred = np.argmax(result)

        if store: # save computation if we're not using backprop
            
            self.hidden_layer_out = first_layer
            self.output = result
            res = np.asarray(first_layer.dot(self.output_weights))
            # if not np.any(res):
            #     raise Exception()
            #     print("first layer",first_layer)
            #     print("weights", self.output_weights)
            self.output_deriv = self.softmax(res.flatten(), target, deriv = True)
            self.first_layer_deriv = self.relu(first_layer_into_node, deriv = True)
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
    start = time.time()
    n_classes_main, n_classes_adv = n_classes
    network = Ann(n_in, n_classes_main, n_hidden, learning_rate, alpha)
    adv = Adv(n_hidden, n_classes_adv, adv_hidden, learning_rate, alpha)
    count = 0
    
    p = round(0.8 * len(train))
    valid_x = train[p:,]
    valid_y = target[:,p:]
    
    train = train[:p,]
    target = target[:, :p]

    
    pred_target, adv_target = target[0], target[1]
    losses = []


    while count < 10:
        print("\n")
        print("Epoch %d" % count)
        error_rate = 0
        
        main_error, adv_error = [], []
        prev_w, prev_out_w = None, None
        for row in range(train.shape[0]):
            # Input x to the network and computer output o_u

            # if row < 10:
            #     if prev_w is not None:
            #         print("\n")
            #         print("Hid w", np.allclose(prev_w, network.weights))
            #         print("Out w", np.allclose(prev_out_w, network.output_weights))
            #     prev_w = network.weights
            #     prev_out_w = network.output_weights

            (logit, main_output) = network.feed_forward(train[row], pred_target[row], store = True)
            main_error.append(main_output)
            #if np.argmax(main_output[1]) == pred_target[row]: main_correct += 1
           
            # update weights of approval network (Ly)
            network.update_weights(np.argmax(main_output), pred_target[row], train[row])

            # get output for race
                # another feed_forward with logit as the input
            #adv_softmax, adv_output = adv.feed_forward(logit, adv_target[row], store = True)
            #adv_error.append(adv_softmax)

            # # calculate the partial for main nn and update the weight for race nn (Ld)
            #adv.update_weights(adv_output, adv_target[row], logit, network, train[row])

        count += 1
        
#         if count in [1,5,10,15]:
#             train_name = "train_nn_%dep" % count
#             adv_name = "adv_nn_%dep" % count
#             pickle.dump(network, open(train_name, "wb"))
#             pickle.dump(adv, open(adv_name, "wb"))
        
        #print("up to logg loss")
        error_rate += log_loss(pred_target, main_error) #- alpha * log_loss(adv_target, adv_error, labels = (0,1,2,3,4)) 
        print("cross entropy loss = " + str(error_rate))
      
        
        corrects = 0
        for row in range(len(valid_x)):
            output = network.feed_forward(valid_x[row], valid_y[0][row])
            if np.argmax(output[1]) == valid_y[0][row]: corrects += 1
        print("Validation accuracy %f" % round(corrects / len(valid_x), 4))
        
        #losses.append(log_loss(pred_target, main_error) - alpha * log_loss(adv_target, adv_error, labels = (0,1,2,3,4)))
    #print("Main Acc", main_train_accs)
    #print("Adv Acc", adv_train_accs)
    print("Total losses", losses)
    
    #df = pd.DataFrame({"main_acc": main_train_accs, "adv_acc" : adv_train_accs, "losses" : losses, "epoch": list(range(1,21))})
    #pd.to_csv(df, "metrics.csv")
    return network, adv, losses

def main():
    '''
    @param: train (str) - path to training data (2d)
    @param: target (str) - path to target data (1d)
    @param: test (str) - path to testing data (2d)
    '''
    ################## TUNING ########################
    n_hidden = 32 # magic number, change to tune neural net
    adv_hidden = 32
    eta = 0.001 # magic number, change to tune neural net
    alpha = 0.1 # tuning param for adversary
    ##################################################
    print("Reading in data")
    x_train = np.load("x_train_data.npy")[:1000,]
    y_train = np.load("y_train_data.npy")[:,:1000]

    x_test = np.load("x_test_data.npy")[:500,]
    y_test = np.load("y_test_data.npy")[:,:500]

    print("Completed reading in data")
    print("Beginning training")
    n_in = x_train.shape[1]
    n_classes = 2, 5#len(set(y_test[0])), len(set(y_test[1]))
    start = time.time()
    trained_nn, trained_adv, losses = train(x_train, y_train, n_in, n_classes, n_hidden, adv_hidden, eta, alpha)
    end = time.time()
    print("Total train time: %d min %d sec" % ((end - start) // 60, (end - start) % 60))
    print("Training losses")
    print(losses)
    sys.stdout.write("TRAINING COMPLETED! NOW PREDICTING.\n")

    corrects = 0
    for row in range(len(x_test)):
        output = trained_nn.feed_forward(x_test[row])
        if np.argmax(output[1]) == y_test[0][row]: corrects += 1
    print("\n")
    print("Test accuracy: %f " % float(corrects / len(x_test)))
    # saving model
#     print("SAVING MODEL")
#     pickle.dump(trained_nn, open("train_nn_20ep", "wb"))
#     pickle.dump(trained_adv, open("trained_adv_20ep", "wb"))
#     print("SAVED MODEL")
main()