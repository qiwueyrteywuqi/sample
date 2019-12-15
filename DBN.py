# -*- coding: utf-8 -*-
"""
Created on Fri Dec 13 19:12:42 2019

@author: User
"""

# Import the math function for calculations
import math
# Tensorflow library. Used to implement machine learning models
import tensorflow as tf
# Numpy contains helpful functions for efficient mathematical calculations
import numpy as np
# Image library for image manipulation
# import Image
# Utils file
# Getting the MNIST data provided by Tensorflow
import csv
import random
import matplotlib.pyplot as plt
import pickle
from sklearn.metrics import classification_report

# Class that defines the behavior of the RBM
class RBM(object):
    def __init__(self, input_size, output_size):
        # Defining the hyperparameters
        self._input_size = input_size  # Size of input
        self._output_size = output_size  # Size of output
        self.epochs = 50  # Amount of training iterations
        self.learning_rate = 1  # The step used in gradient descent
        self.batchsize = 50  # The size of how much data will be used for training per sub iteration

        # Initializing weights and biases as matrices full of zeroes
        self.w = np.zeros([input_size, output_size], np.float32)  # Creates and initializes the weights with 0
        self.hb = np.zeros([output_size], np.float32)  # Creates and initializes the hidden biases with 0
        self.vb = np.zeros([input_size], np.float32)  # Creates and initializes the visible biases with 0

    # Fits the result from the weighted visible layer plus the bias into a sigmoid curve
    def prob_h_given_v(self, visible, w, hb):
        # Sigmoid
        return tf.nn.sigmoid(tf.matmul(visible, w) + hb)

    # Fits the result from the weighted hidden layer plus the bias into a sigmoid curve
    def prob_v_given_h(self, hidden, w, vb):
        return tf.nn.sigmoid(tf.matmul(hidden, tf.transpose(w)) + vb)

    # Generate the sample probability
    def sample_prob(self, probs):
        return tf.nn.relu(tf.sign(probs - tf.random_uniform(tf.shape(probs))))

    # Training method for the model
    def train(self, X):
        # Create the placeholders for our parameters
        _w = tf.placeholder("float", [self._input_size, self._output_size])
        _hb = tf.placeholder("float", [self._output_size])
        _vb = tf.placeholder("float", [self._input_size])

        prv_w = np.zeros([self._input_size, self._output_size],
                         np.float32)  # Creates and initializes the weights with 0
        prv_hb = np.zeros([self._output_size], np.float32)  # Creates and initializes the hidden biases with 0
        prv_vb = np.zeros([self._input_size], np.float32)  # Creates and initializes the visible biases with 0

        cur_w = np.zeros([self._input_size, self._output_size], np.float32)
        cur_hb = np.zeros([self._output_size], np.float32)
        cur_vb = np.zeros([self._input_size], np.float32)
        v0 = tf.placeholder("float", [None, self._input_size])

        # Initialize with sample probabilities
        h0 = self.sample_prob(self.prob_h_given_v(v0, _w, _hb))
        v1 = self.sample_prob(self.prob_v_given_h(h0, _w, _vb))
        h1 = self.sample_prob(self.prob_h_given_v(v1, _w, _hb))

        # Create the Gradients
        positive_grad = tf.matmul(tf.transpose(v0), h0)
        negative_grad = tf.matmul(tf.transpose(v1), h1)
        
        # CD
        CD = (positive_grad - negative_grad) / tf.to_float(tf.shape(v0)[0])
        
        # Update learning rates for the layers
        update_w = _w + self.learning_rate * CD
        update_vb = _vb + self.learning_rate * tf.reduce_mean(v0 - v1, 0)
        update_hb = _hb + self.learning_rate * tf.reduce_mean(h0 - h1, 0)

        # Find the error rate
        err = tf.reduce_mean(tf.square(v0 - v1))
        
        error = []
        # Training loop
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            # For each epoch
            for epoch in range(self.epochs):
                # For each step/batch
                for start, end in zip(range(0, len(X), self.batchsize), range(self.batchsize, len(X), self.batchsize)):
                    batch = X[start:end]
                    # Update the rates
                    cur_w = sess.run(update_w, feed_dict={v0: batch, _w: prv_w, _hb: prv_hb, _vb: prv_vb})
                    cur_hb = sess.run(update_hb, feed_dict={v0: batch, _w: prv_w, _hb: prv_hb, _vb: prv_vb})
                    cur_vb = sess.run(update_vb, feed_dict={v0: batch, _w: prv_w, _hb: prv_hb, _vb: prv_vb})
                    prv_w = cur_w
                    prv_hb = cur_hb
                    prv_vb = cur_vb
                error.append(sess.run(err, feed_dict={v0: X, _w: cur_w, _vb: cur_vb, _hb: cur_hb}))
                print('Epoch: %d' % epoch, 'reconstruction error: %f' % error[-1])
            self.w = prv_w
            self.hb = prv_hb
            self.vb = prv_vb
            
        plt.plot(error)
        plt.xlabel("Batch Number")
        plt.ylabel("Error")
        plt.show()
    # Create expected output for our DBN
    def rbm_outpt(self, X):
        input_X = tf.constant(X)
        _w = tf.constant(self.w)
        _hb = tf.constant(self.hb)
        out = tf.nn.sigmoid(tf.matmul(input_X, _w) + _hb)
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            return sess.run(out)

class NN(object):

    def __init__(self, sizes, X, Y, validationX, validationY):
        # Initialize hyperparameters
        self._sizes = sizes
        self._X = X
        self._Y = Y
        self.vX = validationX
        self.vY = validationY
        self.w_list = []
        self.b_list = []
        self._learning_rate = 1
        self._momentum = 0.9
        self._epoches = 500
        self._batchsize = 100
        input_size = X.shape[1]

        # initialization loop
        for size in self._sizes + [Y.shape[1]]:
            # Define upper limit for the uniform distribution range
            max_range = 4 * math.sqrt(6. / (input_size + size))

            # Initialize weights through a random uniform distribution
            self.w_list.append(
                np.random.uniform(-max_range, max_range, [input_size, size]).astype(np.float32))

            # Initialize bias as zeroes
            self.b_list.append(np.zeros([size], np.float32))
            input_size = size

    # load data from rbm
    def load_from_rbms(self, dbn_sizes, rbm_list):
        # Check if expected sizes are correct
        assert len(dbn_sizes) == len(self._sizes)

        for i in range(len(self._sizes)):
            # Check if for each RBN the expected sizes are correct
            assert dbn_sizes[i] == self._sizes[i]

        # If everything is correct, bring over the weights and biases
        for i in range(len(self._sizes)):
            self.w_list[i] = rbm_list[i].w
            self.b_list[i] = rbm_list[i].hb

    # Training method
    def train(self):
        # Create placeholders for input, weights, biases, output
        _a = [None] * (len(self._sizes) + 2)
        _w = [None] * (len(self._sizes) + 1)
        _b = [None] * (len(self._sizes) + 1)
        _a[0] = tf.placeholder("float", [None, self._X.shape[1]])
        y = tf.placeholder("float", [None, self._Y.shape[1]])

        # Define variables and activation functoin
        for i in range(len(self._sizes) + 1):
            _w[i] = tf.Variable(self.w_list[i])
            _b[i] = tf.Variable(self.b_list[i])
        for i in range(1, len(self._sizes) + 2):
            _a[i] = tf.nn.sigmoid(tf.matmul(_a[i - 1], _w[i - 1]) + _b[i - 1])

        # Define the cost function
        cost = tf.reduce_mean(tf.square(_a[-1] - y))

        # Define the training operation (Momentum Optimizer minimizing the Cost function)
        train_op_Momentum = tf.train.MomentumOptimizer(self._learning_rate, self._momentum).minimize(cost)
        train_op_Adagrad = tf.train.AdagradOptimizer(learning_rate=self._learning_rate).minimize(cost)
        train_op_Adam = tf.train.AdamOptimizer(learning_rate = self._learning_rate, epsilon=1e-08).minimize(cost)

        # Prediction operation
        predict_op = tf.argmax(_a[-1], 1)
        
        acc = []
        val_acc = []
        # Training Loop
        with tf.Session() as sess:
            # Initialize Variables
            sess.run(tf.global_variables_initializer())

            # For each epoch
            for i in range(self._epoches):

                # For each step
                for start, end in zip(
                        range(0, len(self._X), self._batchsize), range(self._batchsize, len(self._X), self._batchsize)):
                    # Run the training operation on the input data
                    sess.run(train_op_Momentum, feed_dict={
                        _a[0]: self._X[start:end], y: self._Y[start:end]})

                for j in range(len(self._sizes) + 1):
                    # Retrieve weights and biases
                    self.w_list[j] = sess.run(_w[j])
                    self.b_list[j] = sess.run(_b[j])
                    
                acc.append(np.mean(np.argmax(self._Y, axis=1) == sess.run(predict_op, feed_dict={_a[0]: self._X, y: self._Y})))
                val_acc.append(np.mean(np.argmax(self.vY, axis=1) == sess.run(predict_op, feed_dict={_a[0]: self.vX, y: self.vY})))
                    
                print("Accuracy rating for epoch " + str(i+1) + ": " + str(np.mean(np.argmax(self._Y, axis=1) == \
                                                                                 sess.run(predict_op, feed_dict={_a[0]: self._X, y: self._Y}))))
                print("Validation rating for epoch " + str(i+1) + ": " + str(np.mean(np.argmax(self.vY, axis=1) == \
                                                                                 sess.run(predict_op, feed_dict={_a[0]: self.vX, y: self.vY}))))
                
                target_names = ['class 1', 'class 2', 'class 3', 'class 4']
                print(classification_report(np.argmax(self.vY, axis=1), sess.run(predict_op, feed_dict={_a[0]: self.vX, y: self.vY}), target_names = target_names))
                
        plt.plot(acc,'b')
        plt.plot(val_acc,'r')
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy Rating")
        plt.legend(['train', 'validation'], loc='center right')
        plt.savefig("DBN_train_history_accuracy.png")
        plt.show()

if __name__ == '__main__':
    # Loading in the data
    
    with open('Energy_Spectrum_1.csv', newline='') as csvfile:
        input_datas = list(csv.reader(csvfile))
    with open('Energy_Spectrum_2.csv', newline='') as csvfile:
        input_datas.extend(csv.reader(csvfile))
    with open('Energy_Spectrum_3.csv', newline='') as csvfile:
        input_datas.extend(csv.reader(csvfile))
    with open('Energy_Spectrum_4.csv', newline='') as csvfile:
        input_datas.extend(csv.reader(csvfile))
        
    decomposed_level = 6

    input_N = len(input_datas)
    input_energy_N = pow(2,decomposed_level)
    input_label = 4
####################################################################
#####Training set and testing set sampling and one-hot encoding#####  
    training_percentage = 0.8
    training_N = int(input_N * training_percentage)
    testing_N = input_N - training_N

    training_data = np.zeros((training_N,input_energy_N))
    training_label = np.zeros((training_N,input_label))
    testing_data = np.zeros((testing_N,input_energy_N))
    testing_label = np.zeros((testing_N,input_label))

    sampling_Training_data = random.sample(input_datas,training_N)
    for k in range(input_label):
        for i in range(training_N): 
            if sampling_Training_data[i][input_energy_N] == str(k+1) :
                for j in range(input_energy_N) :
                    training_data[i,j] = sampling_Training_data[i][j]
                    training_label[i,k] = 1

    sampling_Testing_data = [data for data in input_datas if data not in sampling_Training_data]
    random.shuffle(sampling_Testing_data)
    for k in range(input_label):
        for i in range(testing_N): 
            if sampling_Testing_data[i][input_energy_N] == str(k+1):
                for j in range(input_energy_N):
                    testing_data[i,j] = sampling_Testing_data[i][j]
                    testing_label[i,k] = 1

        
        
        

    RBM_hidden_sizes = [50, 50]  # create 4 layers of RBM with size 785-500-200-50
    # Since we are training, set input as training data
    inpX = training_data.astype(np.float32)
    #inpX = trX
    # Create list to hold our RBMs
    rbm_list = []
    # Size of inputs is the number of inputs in the training set
    input_size = inpX.shape[1]

    # For each RBM we want to generate
    for i, size in enumerate(RBM_hidden_sizes):
        print('RBM: ', i, ' ', input_size, '->', size)
        rbm_list.append(RBM(input_size, size))
        input_size = size

    # For each RBM in our list
    for rbm in rbm_list:
        print('New RBM:')
        # Train a new one
        rbm.train(inpX)
        # Return the output layer
        inpX = rbm.rbm_outpt(inpX)

    nNet = NN(RBM_hidden_sizes, training_data, training_label, testing_data, testing_label)
    #nNet = NN(RBM_hidden_sizes, trX, trY)
    nNet.load_from_rbms(RBM_hidden_sizes, rbm_list)
    nNet.train()
    # Save model
    Pkl_Filename = "DBN_model.pkl"
    with open(Pkl_Filename, 'wb') as file:
        pickle.dump(nNet, file)
        
    