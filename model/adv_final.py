import sys
import numpy as np
import random
import time
import pandas as pd
import pickle
from sklearn.metrics import log_loss


class Ann(object):

    def __init__(self, n_in, n_classes, n_hidden, learning_rate, alpha, weight_decay):
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
        self.weight_decay = weight_decay

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
        res = np.matrix(first_layer) * np.matrix(self.output_weights) # 32x1  32x2 = 1x2
        
        result = np.asarray(res).flatten()
       
        probs = self.softmax(result)

        if store: # save computation if we're not using backprop
            self.hidden_layer_out = first_layer
            self.output_deriv =  self.softmax(result, deriv = True)
            self.first_layer_deriv = self.relu(first_layer_into_node, deriv = True)
            #self.first_layer_deriv[-1] = 1 # We think this is bias
        return (logit, probs)


    def update_weights(self, output, truth, train):
        '''
        @param: output (float) - output of neural net
        @param: truth (str) - supervised label of data, yes or no
        @param: train(1d list) - input values
        @return: None - updates weights of neural net using grad. desc.
        '''
        error = np.zeros(self.n_classes)
        error[int(truth)] = -(1 / (output + 1e-10))
        self.delta_k = np.matrix(error) * np.matrix(self.output_deriv).transpose() # 1x2, 2x2 = 1x2
        self.delta_h = np.matrix(self.first_layer_deriv) * np.matrix(self.output_weights) * np.matrix(self.delta_k).transpose() # 32x32, 32x2, 1x2 = 32x1
        # delta h should be 1 x 32
        
        # update the weights b/w hidden layer and output

        gradient_top = self.learning_rate * np.matrix(self.hidden_layer_out).transpose() * np.matrix(self.delta_k) # 32 x 1, 2x1 = 32x2
        gradient_top = np.clip(gradient_top, -1, 1)

        
        gradient_bottom = self.learning_rate * np.matrix(train).transpose() * np.matrix(self.delta_h).transpose() # 32 x 31x1, 32x1 = 31x32
        gradient_bottom = np.clip(gradient_bottom, -1, 1)

        return gradient_top, gradient_bottom


    def softmax(self, x, deriv = False):
        x_norm = (x - np.min(x)) / (np.max(x) - np.min(x) + 1)
        probs = np.asarray(np.exp(x_norm) / sum(np.exp(x_norm))).flatten()
        
        if deriv: #d cross / d soft * d soft / d relu output, no need for dJ in update_weights
            res = np.zeros((self.n_classes,self.n_classes))
            for i in range(self.n_classes):
                for j in range(self.n_classes):
                    if i == j:
                        res[i,j] = probs[i] * (1 - probs[i])
                    else:
                        res[i,j] = probs[i] * - probs[j]
            return res
        return probs

    def relu(self, x, deriv = False):
        if deriv:
            return np.diagflat(np.where(x > 1, 1, 0))
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

    def __init__(self, n_in, n_classes, n_hidden, learning_rate, alpha, weight_decay):
        super().__init__(n_in, n_classes, n_hidden, learning_rate, alpha, weight_decay)
        self.n_classes = n_classes
        self.alpha = alpha
        self.output_weights = np.random.uniform(-0.05, 0.05, (self.n_out, self.n_classes)) # output weights: n_hidden + 1 x n_classes


    def update_weights(self, output, truth, train, main_nn, main_train):
        # ADV UPDATES 
        
        error = np.zeros(self.n_classes)
        error[int(truth)] = - 1 / output # probability of class truth
        self.delta_k = np.matrix(error) * np.matrix(self.output_deriv) # 1x5
        self.delta_h = np.matrix(self.first_layer_deriv) * np.matrix(self.output_weights) * self.delta_k.transpose()  # 32 x 32, 32x5,  1x5

        # update the weights b/w hidden layer and output
        gradient_top = self.learning_rate * np.matrix(self.hidden_layer_out).transpose() * np.matrix(self.delta_k) # 32x1 , 1x5

        gradient_bottom = self.learning_rate * np.matrix(self.delta_h) * np.matrix(train)#32x32 

        # # MAIN NET UPDATES
        self.delta_q = np.matrix(main_nn.first_layer_deriv) * np.matrix(self.weights) * np.matrix(self.delta_h)
        #32x32, 32x32, 1x32
        grad_btm = np.matrix(self.delta_q) * np.matrix(main_train)
        # 31x1, 32x1
        grad_btm = self.clip(grad_btm)

        # Adv weights update 
        gradient_top = np.clip(gradient_top, -0.01, 0.01)
        gradient_bottom = np.clip(gradient_bottom, -0.01, 0.01)

        return gradient_top, gradient_bottom, grad_btm
         


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
        #if not np.any(first_layer):
        #    raise Exception("weights ", self.weights)


        # the vector that feeds into the activation function
        first_layer_into_node = np.array(data.dot(self.weights)) # 1 x n_hidden + 1
        x = np.asarray(first_layer.dot(self.output_weights))
        result = self.softmax(x.flatten()) # 1 x n_hidden + 1

        pred = np.argmax(result)

        if store: # save computation if we're not using backprop
            
            self.hidden_layer_out = first_layer
            self.output = result
            res = np.asarray(first_layer.dot(self.output_weights))
            self.output_deriv = self.softmax(res.flatten(), deriv = True)
            self.first_layer_deriv = self.relu(first_layer_into_node, deriv = True)
        return result, pred 



def train(train, target, n_in, n_classes, n_hidden, adv_hidden, learning_rate = 0.1, alpha = 1, epochs = 20, batch_size=256, weight_decay = 0.99):
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
    full_data = np.concatenate((train,target.transpose()), axis = 1)

    n_classes_main, n_classes_adv = n_classes
    network = Ann(n_in, n_classes_main, n_hidden, learning_rate, alpha, weight_decay)
    adv = Adv(n_hidden, n_classes_adv, adv_hidden, learning_rate, alpha, weight_decay)
    count = 0
    
    p = round(0.8 * len(train))
    valid = full_data[p:,]

    np.random.shuffle(valid)
    valid_x = valid[:,:-2]
    valid_y = valid[:,-2:].transpose()
    
    
    train_full = full_data[:p,]

    train_losses = []
    valid_losses = []
    main_acc, adv_acc = [],[]
    
    while count < epochs:
        print("\n")
        print("Epoch: %d" % count)
        start = time.time()
        np.random.shuffle(train_full)
        true_approval = train_full[:,-2]
        true_race = train_full[:,-1]

        num_ones = 0
        num_correct_deny = 0
        n_batches = train_full.shape[0] // batch_size
        batches = []
        for i in range(n_batches +1):
            if (i+1) * batch_size < train_full.shape[0]:
                batches.append(train_full[i * batch_size : (i+1) * batch_size,:])
            else:
                last_batch = train_full[i*batch_size:,]
                if len(last_batch) != 0:
                    batches.append(last_batch)
        
        
        main_error, adv_error = [], []
        prev_w, prev_out_w = None, None
        main_correct, adv_correct = 0,0
        for batch in (batches):

            train = batch[:,:-2]
            target = batch[:,-2:].transpose()
            pred_target, adv_target = target[0], target[1]
            error_rate = 0

            total_top, total_btm, adv_top_all, adv_btm_all = 0,0,0,0

            for row in range(train.shape[0]):
                # Input x to the network and computer output o_u

                (logit, main_output) = network.feed_forward(train[row], pred_target[row], store = True)
                #logit = self.sigmoid(logit)
               
                if len(main_output) != 2: # make sure softmax outputs correct size
                    raise Exception("Main output length not 2")
                       
                # update weights of approval network (Ly)
                main_top, main_bottom = network.update_weights(np.argmax(np.asarray(main_output).flatten()), main_output[1], train[row])

                # get output for race,another feed_forward with logit as the input
                adv_softmax, adv_output = adv.feed_forward(logit, adv_target[row], store = True)

                if len(adv_softmax) != 5: # make sure softmax outputs correct size
                    raise Exception("Adversary output length not 5")
                
                # # calculate the partial for main nn and update the weight for race nn (Ld)
                adv_top, adv_btm, g_btm = adv.update_weights(adv_softmax[int(pred_target[row])], adv_target[row], logit, network, train[row])
                
                if type(g_btm) == int:
                    raise Exception("Adversary update does not update g_btm, g_btm is and integer and not a matrix")

                # store gradients
                total_top += main_top  
                total_btm += main_bottom - network.alpha * g_btm.transpose()
                adv_top_all += adv_top 
                adv_btm_all += adv_btm
                
                # store accuracies and softmax outputs
                if np.argmax(main_output) == pred_target[row]: 
                    main_correct += 1
                if np.argmax(main_output) == 1:
                    num_ones +=1
                if adv_output == adv_target[row]: 
                    adv_correct += 1
                main_error.append(main_output)
                adv_error.append(adv_softmax)
                if pred_target[row] == 0 and np.argmax(main_output) == 0:
                    num_correct_deny +=1

            #prev_weights = network.weights
            network.weights = (network.weights + total_btm) * network.weight_decay 
            network.output_weights = (network.output_weights +  total_top) * network.weight_decay
            adv.weights = (adv.weights + adv_btm_all.transpose()) * network.weight_decay
            adv.output_weights = (adv.output_weights + adv_top_all) * network.weight_decay

        train_end = time.time()
        print("Total training time: %d" % round(train_end - start))
        count += 1
        
        error_rate += log_loss(true_approval, main_error, labels = (0,1)) - alpha * log_loss(true_race, adv_error, labels = (0,1,2,3,4)) 
        print("cross entropy loss = " + str(error_rate))
      
        train_losses.append(error_rate)

        #corrects = 0
        valid_error = []
        valid_start = time.time()
        valid_num_correct_deny, prop_one = 0,0
        for row in range(len(valid_x)):
            _, output = network.feed_forward(valid_x[row], valid_y[0][row])
            valid_error.append(output)
            if valid_y[0][row] == 0 and np.argmax(output) == 0:
                valid_num_correct_deny +=1
            prop_one += np.argmax(output)
        error = log_loss(valid_y[0], valid_error)
        print("Validation cross entropy %f" % round(error, 4))
        valid_end = time.time()
        print("Validation time: %d" % round(valid_end - valid_start))
        valid_losses.append(error)
        
        main_acc.append(main_correct / true_approval.shape[0])
        adv_acc.append(adv_correct / true_race.shape[0])
        print("Main accuracy %f" % (main_correct / true_approval.shape[0]))
        print("Adv accuracy %f" % (adv_correct / true_race.shape[0]))
        
        print("Training, Main proportion of 1s %f" % (num_ones / true_approval.shape[0]))
        print("Training, Num correct deny %d out of %d" % (num_correct_deny, true_approval.shape[0] - sum(true_approval)))
        #print(prop_one)
        #print(valid_y[0].shape[0])
        print("Validation, Main proportion of 1s %f" % (prop_one/valid_y[0].shape[0]))
        print("Validation, Num correct deny %d out of %d" % (valid_num_correct_deny, valid_y[0].shape[0] - sum(valid_y[0])))
    
    acc_df = pd.DataFrame({"main" : main_acc, "adv" :adv_acc, "epoch" : list(range(epochs))})
    acc_df.to_csv("adv_accuracy.csv")
        
    
    losses_df = pd.DataFrame({"train":train_losses, "valid": valid_losses, "epochs" : list(range(epochs))})
    losses_df.to_csv("adv_loss.csv")
    return network, adv

def main():
    '''
    @param: train (str) - path to training data (2d)
    @param: target (str) - path to target data (1d)
    @param: test (str) - path to testing data (2d)
    '''
    ################## TUNING ########################
    n_hidden = 128 # magic number, change to tune neural net
    adv_hidden = 128
    eta = 0.001 # magic number, change to tune neural net
    alpha = 0 # tuning param for adversary
    epochs = 6
    batch_size = 512
    weight_decay = 1.01
    ##################################################
    print("Reading in data")
    nrow = 100000
    x_train = np.load("x_train_data.npy")[:nrow,:]
    y_train = np.load("y_train_data.npy")[:,:nrow]

    x_test = np.load("x_test_data.npy")[nrow//2:,]
    y_test = np.load("y_test_data.npy")[:,nrow//2:]


    print("Completed reading in data")
    print("Beginning training")
    n_in = x_train.shape[1]
    n_classes = len(set(y_test[0])), len(set(y_test[1]))
    
    start = time.time()
    trained_nn, trained_adv = train(x_train, y_train, n_in, n_classes, n_hidden, adv_hidden, eta, alpha, epochs, batch_size, weight_decay)
    end = time.time()
    print("Total train time: %d min %d sec" % ((end - start) // 60, (end - start) % 60))
    sys.stdout.write("TRAINING COMPLETED! NOW PREDICTING.\n")

    test_errors = []
    corrects = 0
    for row in range(len(x_test)):
        _, output = trained_nn.feed_forward(x_test[row], y_test[0][row])
        test_errors.append(output)
        if np.argmax(output) == y_test[0][row]: corrects += 1

    test_error_rate = log_loss(y_test[0], test_errors, labels = [1,0])
    print("\n")
    print("Test cross entropy: %f " % test_error_rate)
    print("Test accuracy: %f" % round(corrects / x_test.shape[0], 4))
    # saving model
    print("SAVING MODEL")
    pickle.dump(trained_nn, open("adv_train_nn", "wb"))
    pickle.dump(trained_adv, open("adv_trained_adv", "wb"))
    print("SAVED MODEL")
    
    
main()