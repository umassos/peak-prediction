#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 23 15:14:50 2019
@author: Akhil
"""
# Libraries to run the code
from numpy.random import seed
seed(1)
from tensorflow import set_random_seed
set_random_seed(2)
from pandas import read_csv
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder 
from sklearn.preprocessing import OneHotEncoder
from pandas import DataFrame
from pandas import concat
import numpy as np
from numpy import concatenate
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.callbacks import EarlyStopping
from keras import regularizers
from keras.optimizers import Adam
#for saving to csv
from os import listdir
import csv
import pandas
from matplotlib import pyplot
import pickle
from math import sqrt
from sklearn.metrics import mean_squared_error
import time
import pandas as pd

# STEP 1
# Data should be of the form as follows: DateTime, Demand, temperature, humidity, Hour of the day, Seasons, Campus schedule
# Keras requires the data to be converted to be in the following format [Samples,Time steps,Features]. Season, hour of day 
# and campus schedule are one hot encoded. This is done using the function below.

def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
	n_vars = 1 if type(data) is list else data.shape[1]
	df = DataFrame(data)
	cols, names = list(), list()
	# Train_X (t-n, ... t-1)
	for i in range(n_in, 0, -1):
		cols.append(df.shift(i))
		names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
	# Train_y (t, t+1, ... t+n)
	for i in range(0, n_out):
		cols.append(df.shift(-i))
		if i == 0:
			names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
		else:
			names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
	# put it all together
	agg = concat(cols, axis=1)
	agg.columns = names
	# drop rows with NaN values
	if dropnan:
		agg.dropna(inplace=True)
	return agg

# All data is normalized between 0 and 1 by function below. 
def scaling(data):
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler(feature_range=(0, 1))
    data = scaler.fit_transform(data)
    data = pd.DataFrame(data)
    return data

def testxy_2FH(data, FH = 24):
    t1 = []
    i = 0
    while i <len(data):
        tt = data[i:i+1]
        t1.append(tt)
        i+=FH
    return t1 
# The scaled data is called in the function below to be split into train and test dataset. Train_X and Train_y are the training 
# dataset and Test_X and Test_y are the test dataset. n_lookback points to the lookback and n_lookahead points to the forecasting 
# horizon.

def traintest(data,lend = 7008, n_fea = 40, n_lookback = 48, n_lookahead = 24):
    ts = series_to_supervised(data,n_lookback,n_lookahead)
    train = ts[:lend]
    test = ts[lend:]   
# Train X
    train_X = train.iloc[:,0:n_lookback * n_fea]
    train_X = np.asarray(train_X)
    train_X = train_X.reshape((train_X.shape[0],n_lookback,n_fea))
# Train y    
    train_y = train.iloc[:,n_lookback * n_fea:]
    train_y = np.asarray(train_y)
# Test
    test = np.asarray(test)
    test = testxy_2FH(test, FH = n_lookahead)
    test = np.concatenate(test,axis = 0)

# Test X 
    test_X = test[:,:n_fea*n_lookback]
    test_X = test_X.reshape((test_X.shape[0],n_lookback,n_fea))
# Test y
    test_y = test[:,n_fea*n_lookback:] 
    test_y = test_y[:,range(0,n_lookahead*n_fea,n_fea)]
    return train_X, train_y, test_X, test_y

# STEP 2

def NN(Neurons = 100, Neuron2 = 125, Neuron3 = 125, Neuron4 = 100, epochs = 50,batchsize = 512,
       breg = 0.01, dropout = 0.1, layer = 2, lr = 0.001, n_lookahead = 24,recurrent = 'hard_sigmoid'):
    global Pickled_save, train_X,train_y,test_y,test_X
    n_lookahead = 24
    fea = '40_Features'
    RS = False
    f_bias = 0
    bs = 1
    print('Started')
    print(Neurons,Neuron2,Neuron3,Neuron4,batchsize,breg,dropout,layer,n_lookahead)
    #
    model=Sequential()
    if layer == 1:
        model.add(LSTM(Neurons,recurrent_activation = 'hard_sigmoid', input_shape=(train_X.shape[1], train_X.shape[2]),kernel_initializer='glorot_uniform', bias_regularizer = regularizers.l2(breg),
               dropout = dropout, return_sequences=False, unit_forget_bias = 1))
    elif layer == 2:
        model.add(LSTM(Neurons,input_shape=(train_X.shape[1], train_X.shape[2]),kernel_initializer='glorot_uniform', bias_regularizer = regularizers.l2(breg),
               dropout = dropout, return_sequences=True,unit_forget_bias = 1))
        model.add(LSTM(Neuron2))
    elif layer == 3:
        model.add(LSTM(Neurons,input_shape=(train_X.shape[1], train_X.shape[2]),kernel_initializer='glorot_uniform', bias_regularizer = regularizers.l2(breg),
                return_sequences=True,unit_forget_bias = 1))
        model.add(LSTM(Neuron2, return_sequences = True,dropout = dropout))
        model.add(LSTM(Neuron3,dropout = dropout))
    elif layer == 4:
        model.add(LSTM(Neurons,input_shape=(train_X.shape[1], train_X.shape[2]),kernel_initializer='glorot_uniform', bias_regularizer = regularizers.l2(breg),
                return_sequences=True,recurrent_activation = recurrent, return_state = RS,unit_forget_bias = f_bias, use_bias = bs))
        model.add(LSTM(Neuron2, dropout = dropout, return_sequences = True,recurrent_activation = recurrent,return_state = RS,unit_forget_bias = f_bias, use_bias = bs))
        model.add(LSTM(Neuron3, dropout = dropout, return_sequences = True,recurrent_activation = recurrent,return_state = RS,unit_forget_bias = f_bias, use_bias = bs))
        model.add(LSTM(Neuron4,recurrent_activation = recurrent, use_bias = bs))
    model.add(Dense(n_lookahead))
    adam = Adam(lr = lr,decay = 0.001)
    model.compile(loss='mean_squared_error', optimizer = adam)
    start = time.time()
    history = model.fit(train_X, train_y, epochs=epochs, batch_size=batchsize,
            validation_split = 0.1, verbose=1, shuffle=True)  
    end = time.time()
    print('The time elapsed for 10 epochs in seconds is', end-start)
    yhat = model.predict(test_X)
    #
    model.get_config()
    k = model.get_weights()
    model.summary() 
    GG = "Model.pickle"
    pickle_out = open(GG,"wb")
    pickle.dump(k,pickle_out)
    #
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    #
    return yhat, loss, val_loss


train_X = np.load('trainX.npy')
train_y = np.load('trainY.npy')
test_X = np.load('testX.npy')
test_y = np.load('testy.npy')


NNTraining0 = NN(Neurons = 100, 
                Neuron2 = 90, 
                Neuron3 = 80,
                Neuron4 = 70,
                epochs = 1,
                batchsize = 72, 
                breg = 0.04, 
                dropout = 0.1, 
                layer = 4, 
                lr = 0.001,
                recurrent = 'hard_sigmoid')

Forecast = NNTraining0[0]
