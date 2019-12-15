# -*- coding: utf-8 -*-
"""
Created on Tue Nov 12 09:44:03 2019

@author: Allen
"""
#import os
import tensorflow as tf
import numpy as np
import random
import csv
#import math
import matplotlib.pyplot as ma
import keras
#from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense,Dropout,Activation,Flatten
from keras.layers import Conv1D,MaxPooling1D,ZeroPadding1D,normalization
#import pywt
#import itertools


def show_train_history(train_history, train, validation):
    ma.figure().set_size_inches(10, 6)
    ma.plot(train_history.history[train],'b',linewidth=1.5)
    ma.plot(train_history.history[validation],'r',linewidth=1.5)
    ma.xticks(fontsize=8)
    ma.yticks(fontsize=8)
    ma.title("Train " + train + " curve",fontsize=18)
    ma.ylabel(train)
    ma.xlabel('Epoch')
    ma.legend(['train', 'validation'], loc='center right')
    ma.savefig("train_history_" + train + ".png")
    ma.show()


with open('Energy_Spectrum_1.csv', newline='') as csvfile:
    input_datas = list(csv.reader(csvfile))
with open('Energy_Spectrum_2.csv', newline='') as csvfile:
    input_datas.extend(csv.reader(csvfile))
with open('Energy_Spectrum_3.csv', newline='') as csvfile:
    input_datas.extend(csv.reader(csvfile))
with open('Energy_Spectrum_4.csv', newline='') as csvfile:
    input_datas.extend(csv.reader(csvfile))
"""

with open('Energy_Spectrum_1.csv', newline='') as csvfile:
    input_datas = list(csv.reader(csvfile))
with open('Energy_Spectrum_2.csv', newline='') as csvfile:
    input_datas.extend(csv.reader(csvfile))
with open('Energy_Spectrum_3.csv', newline='') as csvfile:
    input_datas.extend(csv.reader(csvfile))
with open('Energy_Spectrum_4.csv', newline='') as csvfile:
    input_datas.extend(csv.reader(csvfile))
"""
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

learning_rate = 0.0000001
NumX = pow(2,decomposed_level)
NumY = input_label

#################################
## CLASS MEAN SCATTER CRITERIA ##
C = NumY
K = NumX
M = training_N
Mi = np.zeros(C)
pi = np.zeros(C)
Yi = np.zeros((C,K))
Y = np.zeros(K)
Ri = np.zeros((C,K))
Rc = np.zeros(K)
J = np.zeros(K)

for i in range(C):
    for j in range(M):
        if( training_label[j][i] == 1 ):
            Mi[i] += 1
for i in range(C):
    pi[i] = Mi[i] / M
for i in range(C):
    for j in range(K):
        for k in range(M):
            if( training_label[k][i] == 1 ):
                Yi[i][j] += training_data[k][j]
        Yi[i][j] = Yi[i][j] / Mi[i]
        for k in range(M):
            if( training_label[k][i] == 1 ):
                Ri[i][j] += ( training_data[k][j] - Yi[i][j] )**2
        Ri[i][j] = Ri[i][j] / Mi[i]
for i in range(K):
    for j in range(C):
        Y[i] += Yi[j][i] * pi[j]
    for j in range(C):
        Rc[i] += ( Yi[j][i] - Y[i] )**2 * pi[j]
    for j in range(C):
        J[i] += Ri[j][i]
    if ( J[i] == 0 ):
        J[i] = 0.000001
    J[i] = Rc[i] / J[i]      

CSMC_top = NumX
sort_J = np.argsort(-J)
avg_J = J[sort_J[-CSMC_top:]].sum() / CSMC_top
threshold = J[sort_J[-CSMC_top]]
Selec = 64
#a = training_data
#print(sort_J[-CSMC_top:])
training_data = training_data[:,sort_J[-Selec:]]
testing_data = testing_data[:,sort_J[-Selec:]]

#####################################
################ DNN ################
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.4)  
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
tf.keras.backend.set_session(sess)

model = Sequential()
model.add(Dense(10,input_dim = Selec,activation="relu",kernel_regularizer=keras.regularizers.l2(3e-5)))
model.add(normalization.BatchNormalization(axis=1, momentum=0.9, epsilon=0.001, center=True, scale=False))
model.add(Dropout(rate=0.22))
model.add(Dense(10,activation="relu",kernel_regularizer=keras.regularizers.l2(3e-5)))
model.add(normalization.BatchNormalization(axis=1, momentum=0.9, epsilon=0.001, center=True, scale=False))
model.add(Dropout(rate=0.22))
#model.add(Dense(64,activation="relu",kernel_regularizer=keras.regularizers.l2(3e-5)))
#model.add(normalization.BatchNormalization(axis=1, momentum=0.9, epsilon=0.001, center=True, scale=False))
#model.add(Dropout(rate=0.22))
model.add(Dense(NumY,activation="softmax",kernel_regularizer=keras.regularizers.l2(2e-5)))

print (model.summary())

adam = keras.optimizers.Adam(lr=learning_rate,decay=0, beta_1=0.9, beta_2=0.999, epsilon=1e-08)

model.compile(loss="categorical_crossentropy",optimizer='adam',metrics=['accuracy'])

ES1 = keras.callbacks.EarlyStopping(monitor='val_loss', patience=30, verbose=1, mode='min')
ES2 = keras.callbacks.EarlyStopping(monitor='val_acc', patience=40, verbose=1, mode='max')
train_history = model.fit(training_data,training_label,validation_data=(testing_data,testing_label),epochs=200,batch_size=32 ,shuffle=True,verbose=1,callbacks=[ES1,ES2])

model.save("DNN.h5")

show_train_history(train_history,'accuracy','val_accuracy')
show_train_history(train_history,'loss','val_loss')



"""
class DNN:
    
    def __init__(self, n_features, n_labels, learning_rate=0.1, n_hidden=1000, activation=tf.nn.relu,
               dropout_ratio=0.5, alpha=0):
        self.n_features = n_features
        self.n_labels = n_labels
        self.weights = None
        self.biases = None
        
        self.graph = tf.Graph()
        self.build(learning_rate, n_hidden, activation, dropout_ratio, alpha)
        self.sess = tf.Session(graph = self.graph)
    
    def build(self, learning_rate, n_hidden, activation, dropout_ratio, alpha):
        with self.graph.as_defaulf():
            
            self.train_features = tf.placeholder(tf.float32, shape=(None, self.n_features))
            self.train_labels = tf.placeholder(tf.int32, shape=(None, self.n_labels))
            
            ### Optimalization
            # build neurel network structure and get their predictions and loss
            self.y_, self.original_loss = self.structure(features=self.train_features,
                                                         labels=self.train_labels,
                                                         n_hidden=n_hidden,
                                                         activation=activation,
                                                         dropout_ratio=dropout_ratio,
                                                         train=True)
            # regularization loss
            self.regularization = \
                tf.reduce_sum([tf.nn.l2_loss(w) for w in self.weights.values()]) \
                / tf.reduce_sum([tf.size(w, out_type=tf.float32) for w in self.weights.values()])

            # total loss
            self.loss = self.original_loss + alpha * self.regularization

            # define training operation
            optimizer = tf.train.GradientDescentOptimizer(learning_rate)
            self.train_op = optimizer.minimize(self.loss)

            ### Prediction
            self.new_features = tf.placeholder(tf.float32, shape=(None, self.n_features))
            self.new_labels = tf.placeholder(tf.int32, shape=(None, self.n_labels))
            self.new_y_, self.new_original_loss = self.structure(features=self.new_features,
                                                                 labels=self.new_labels,
                                                                 n_hidden=n_hidden,
                                                                 activation=activation)
            self.new_loss = self.new_original_loss + alpha * self.regularization

            ### Initialization
            self.init_op = tf.global_variables_initializer()

    def structure(self, features, labels, n_hidden, activation, dropout_ratio=0, train=False):
        # build neurel network structure and return their predictions and loss
        ### Variable
        if (not self.weights) or (not self.biases):
            self.weights = {
                'fc1': tf.Variable(tf.truncated_normal(shape=(self.n_features, n_hidden))),
                'fc2': tf.Variable(tf.truncated_normal(shape=(n_hidden, self.n_labels))),
            }
            self.biases = {
                'fc1': tf.Variable(tf.zeros(shape=(n_hidden))),
                'fc2': tf.Variable(tf.zeros(shape=(self.n_labels))),
            }
        ### Structure
        # layer 1
        fc1 = self.get_dense_layer(features, self.weights['fc1'],
                                   self.biases['fc1'], activation=activation)
        if train:
            fc1 = tf.nn.dropout(fc1, keep_prob=1-dropout_ratio)

        # layer 2
        logits = self.get_dense_layer(fc1, self.weights['fc2'], self.biases['fc2'])

        y_ = tf.nn.softmax(logits)

        loss = tf.reduce_mean(
                 tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits))

        return (y_, loss)

    def get_dense_layer(self, input_layer, weight, bias, activation=None):
        # fully connected layer
        x = tf.add(tf.matmul(input_layer, weight), bias)
        if activation:
            x = activation(x)
        return x

    def fit(self, X, y, epochs=10, validation_data=None, test_data=None, batch_size=None):
        X = self._check_array(X)
        y = self._check_array(y)

        N = X.shape[0]
        random.seed(9000)
        if not batch_size:
            batch_size = N

        self.sess.run(self.init_op)
        for epoch in range(epochs):
            print('Epoch %2d/%2d: ' % (epoch+1, epochs))

            # mini-batch gradient descent
            index = [i for i in range(N)]
            random.shuffle(index)
            while len(index) > 0:
                index_size = len(index)
                batch_index = [index.pop() for _ in range(min(batch_size, index_size))]

                feed_dict = {
                    self.train_features: X[batch_index, :],
                    self.train_labels: y[batch_index],
                }
                _, loss = self.sess.run([self.train_op, self.loss], feed_dict=feed_dict)

                print('[%d/%d] loss = %9.4f     ' % (N-len(index), N, loss), end='\r')

            # evaluate at the end of this epoch
            y_ = self.predict(X)
            train_loss = self.evaluate(X, y)
            train_acc = self.accuracy(y_, y)
            msg = '[%d/%d] loss = %8.4f, acc = %3.2f%%' % (N, N, train_loss, train_acc*100)

            if validation_data:
                val_loss = self.evaluate(validation_data[0], validation_data[1])
                val_acc = self.accuracy(self.predict(validation_data[0]), validation_data[1])
                msg += ', val_loss = %8.4f, val_acc = %3.2f%%' % (val_loss, val_acc*100)

            print(msg)

        if test_data:
            test_acc = self.accuracy(self.predict(test_data[0]), test_data[1])
            print('test_acc = %3.2f%%' % (test_acc*100))

    def accuracy(self, predictions, labels):
        return (np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))/predictions.shape[0])

    def predict(self, X):
        X = self._check_array(X)
        return self.sess.run(self.new_y_, feed_dict={self.new_features: X})

    def evaluate(self, X, y):
        X = self._check_array(X)
        y = self._check_array(y)
        return self.sess.run(self.new_loss, feed_dict={self.new_features: X,
                                                       self.new_labels: y})

    def _check_array(self, ndarray):
        ndarray = np.array(ndarray)
        if len(ndarray.shape) == 1:
            ndarray = np.reshape(ndarray, (1, ndarray.shape[0]))
        return ndarray


if __name__ == '__main__':
    print('Extract MNIST Dataset ...')

    mnist = input_data.read_data_sets('MNIST_data/', one_hot=True)

    train_data = mnist.train
    valid_data = mnist.validation
    test_data = mnist.test

    model = DNNLogisticClassification(
        n_features=28*28,
        n_labels=10,
        learning_rate=0.5,
        n_hidden=1000,
        activation=tf.nn.relu,
        dropout_ratio=0.5,
        alpha=0.01,
    )
    model.fit(
        X=train_data.images,
        y=train_data.labels,
        epochs=3,
        validation_data=(valid_data.images, valid_data.labels),
        test_data=(test_data.images, test_data.labels),
        batch_size=32,
    )
    """