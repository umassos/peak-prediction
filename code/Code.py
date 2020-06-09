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
from keras.wrappers.scikit_learn import KerasRegressor
from os import listdir
import csv
import pandas
from matplotlib import pyplot
import pickle
from math import sqrt
from sklearn.metrics import mean_squared_error
import time
import pandas as pd
import matplotlib.pyplot as plt

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

# The scaled data is called in the function below to be split into train and test dataset. Train_X and Train_y are the training 
# dataset and Test_X and Test_y are the test dataset. n_lookback points to the lookback and n_lookahead points to the forecasting 
# horizon.

def traintest(data, n_fea = 40, n_lookback = 48, n_lookahead = 24):
    ts = series_to_supervised(data,n_lookback,n_lookahead)  
# Train X
    train_X = train.iloc[:,0:n_lookback * n_fea]
    train_X = np.asarray(train_X)
    train_X = train_X.reshape((train_X.shape[0],n_lookback,n_fea))
    np.save(train_X, train_X)
# Train y    
    train_y = train.iloc[:,n_lookback * n_fea:]
    train_y = np.asarray(train_y)
    np.save(train_y, train_y)
# Test X 
    test_X = test[:,:n_fea*n_lookback]
    test_X = test_X.reshape((test_X.shape[0],n_lookback,n_fea))
    np.save(test_X,test_X)
# Test y
    test_y = test[:,n_fea*n_lookback:] 
    test_y = test_y[:,range(0,n_lookahead*n_fea,n_fea)]
    np.save(test_y,test_y)
    return train_X, train_y, test_X, test_y

# STEP 2
# The function below calls the processed dataset and trains the neural network. the function nn_model defines the LSTM model
# based on user settings. 

def nn_model(Neuron1, Neuron2, Neuron3, Neuron4, epochs, batchsize,
       breg, dropout, layer, n_lookahead ,recurrent):
    #
    model=Sequential()
    if layer == 1:
        model.add(LSTM(Neuron1,recurrent_activation = 'hard_sigmoid', input_shape=(train_X.shape[1], train_X.shape[2]),kernel_initializer='glorot_uniform', bias_regularizer = regularizers.l2(breg),
               dropout = dropout, return_sequences=False, unit_forget_bias = 1))
    elif layer == 2:
        model.add(LSTM(Neuron1,input_shape=(train_X.shape[1], train_X.shape[2]),kernel_initializer='glorot_uniform', bias_regularizer = regularizers.l2(breg),
               dropout = dropout, return_sequences=True,unit_forget_bias = 1))
        model.add(LSTM(Neuron2))
    elif layer == 3:
        model.add(LSTM(Neuron1,input_shape=(train_X.shape[1], train_X.shape[2]),kernel_initializer='glorot_uniform', bias_regularizer = regularizers.l2(breg),
                return_sequences=True,unit_forget_bias = 1))
        model.add(LSTM(Neuron2, return_sequences = True,dropout = dropout))
        model.add(LSTM(Neuron3,dropout = dropout))
    elif layer == 4:
        model.add(LSTM(Neuron1,input_shape=(train_X.shape[1], train_X.shape[2]),kernel_initializer='glorot_uniform', bias_regularizer = regularizers.l2(breg),
                return_sequences=True,recurrent_activation = recurrent,unit_forget_bias = 0, use_bias = 1))
        model.add(LSTM(Neuron2, dropout = dropout, return_sequences = True,recurrent_activation = recurrent,unit_forget_bias = 0, use_bias = 1))
        model.add(LSTM(Neuron3, dropout = dropout, return_sequences = True,recurrent_activation = recurrent,unit_forget_bias = 0, use_bias = 1))
        model.add(LSTM(Neuron4,recurrent_activation = recurrent, use_bias = 1))
    model.add(Dense(n_lookahead))
    adam = Adam(lr = 0.1,decay = 0.001)
    model.compile(loss='mean_squared_error', optimizer = adam)
    return model

# Once the model is defined, the hyperparameters of the neural network can be tuned in the user defined grid space using the
# code below. The number of layers, number of neurons, regularizer and dropout are the hyperparameters tuned and its grid 
# space is defined by the values in between a user defined range at regular steps. This means that the user defines the range
# of neurons by definining (Nstart, Nstop, Nstep) in the code below. Nstart = min value of range, Nstop = Max value of range, 
# Nstep = steps/interval length to generate the values between start and stop. The same translate to regularizer (Bstart, Bstop,
# Bstep) and dropout (Dstart, Dstop, Dstep).

def gridsearch(Nstart,Nstop,Nstep,Bstart, Bstop, Bstep,Dstart, Dstop, Dstep, layer):
    Neuron1 = np.arange(Nstart,Nstop,Nstep)
    Neuron2 = np.arange(Nstart,Nstop,Nstep)
    Neuron3 = np.arange(Nstart,Nstop,Nstep)
    Neuron4 = np.arange(Nstart,Nstop,Nstep)
    breg = np.arange(Bstart, Bstop, Bstep)
    dropout = np.arange(Dstart, Dstop, Dstep)
    reg = KerasRegressor(build_fn = nn_model, epochs = 100, verbose = 1, batchsize = 72, layer = layer, n_lookahead = 24, recurrent = 'hard_sigmoid')
    param_grid = dict(Neuron1 = Neuron1, Neuron2 = Neuron2,
                      Neuron3 = Neuron3, Neuron4 = Neuron4, breg = breg, dropout = dropout)
    grid = GridSearchCV(reg, param_grid=param_grid, n_jobs=-1)
    fit_model = grid.fit(train_X, train_y)
    pred = fit_model.predict(test_X)
    print(model.best_score_)
    print(model.best_params_)
    return fit_model, pred
