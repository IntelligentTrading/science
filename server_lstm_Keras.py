import sys
import time
import numpy as np
import pandas as pd
import os
import matplotlib
import matplotlib.pyplot as plt
import random
from matplotlib.dates import DateFormatter, WeekdayLocator, DayLocator, MONDAY
import mysql.connector
from mysql.connector import errorcode

import tensorflow as tf
import shutil
import tensorflow.contrib.learn as tflearn
import tensorflow.contrib.layers as tflayers
from tensorflow.contrib.learn.python.learn import learn_runner
import tensorflow.contrib.metrics as metrics
import tensorflow.contrib.rnn as rnn

from keras.models import load_model, Model, Sequential
from keras.layers import Dense, Activation, Dropout, Input, LSTM, Reshape, Lambda, RepeatVector
from keras.initializers import glorot_uniform
from keras.utils import to_categorical
from keras.optimizers import Adam, adagrad
from keras import backend as K
from keras.callbacks import CSVLogger

from ittutils import ittconnection, get_resampled, get_raw_price, get_raw_volume, classification_dataset_from_ts

print(tf.__version__)
print(sys.version)

db_connection = ittconnection('prodcopy')

COINS_LIST = [('BTC', 2), ('ETH', 2), ('ETH', 0), ("ETC", 0), ('OMG', 0), ('XRP', 0), ('XMR', 0), ('LTC', 0),
              ('XEM', 0)]

X_train = []
Y_train = []

# generate datasets
for transaction_coin, counter_coin in COINS_LIST:
    raw_price_ts = get_raw_price(db_connection, transaction_coin, counter_coin)
    raw_volume_ts = get_raw_volume(db_connection, transaction_coin, counter_coin)

    raw_data_frame = pd.merge(raw_price_ts, raw_volume_ts, how='left', left_index=True, right_index=True)
    print('Shape of ' + transaction_coin + str(raw_data_frame.shape))

    raw_data_frame[pd.isnull(raw_data_frame)] = None

    # resample (for smoothing) and normalize (for learning)
    res_period = '10min'
    data_ts = raw_data_frame.resample(rule=res_period).mean()
    data_ts['price_var'] = raw_data_frame['price'].resample(rule=res_period).var()
    data_ts['volume_var'] = raw_data_frame['volume'].resample(rule=res_period).var()

    data_ts = data_ts.interpolate()

    win_size = 200  # do prediction based on history
    future = 90  # predict a price return in future timepoints (6h?)
    delta = 0.02  # consider UP if last price has been changed by more then delta persents

    if transaction_coin == 'BTC':
        X_test, Y_test, y_tst_price = classification_dataset_from_ts(data_df=data_ts, win_size=win_size, stride=1,
                                                                     future=future, delta=delta)
    else:
        X_train_one, Y_train_one, y_tr_price = classification_dataset_from_ts(data_df=data_ts, win_size=win_size,
                                                                              stride=1, future=future, delta=delta)

        if X_train == []:
            X_train = X_train_one
            Y_train = Y_train_one
        else:
            X_train = np.concatenate((X_train, X_train_one), axis=0)
            Y_train = np.concatenate((Y_train, Y_train_one), axis=0)

# delete all examples with NaN inside
idx2delete = []
for n in range(X_train.shape[0] - 1):
    if np.isnan(X_train[n, :, :]).any():
        idx2delete.append(n)

X_train = np.delete(X_train, (idx2delete), axis=0)
Y_train = np.delete(Y_train, (idx2delete), axis=0)

###### NORMALIZE

for example in range(X_train.shape[0]):
    X_train[example,:,0] = (X_train[example,:,0] - X_train[example,-1,0]) / (np.max(X_train[example,:,0]) - np.min(X_train[example,:,0]))
    X_train[example,:,1] = (X_train[example,:,1] - X_train[example,-1,1]) / (np.max(X_train[example,:,1]) - np.min(X_train[example,:,1]))
    X_train[example,:,2] = (X_train[example,:,2] - X_train[example,-1,2]) / (np.max(X_train[example,:,2]) - np.min(X_train[example,:,2]))
    X_train[example,:,3] = (X_train[example,:,3] - X_train[example,-1,3]) / (np.max(X_train[example,:,3]) - np.min(X_train[example,:,3]))


for example in range(X_test.shape[0]):
    X_test[example,:,0] = (X_test[example,:,0] - X_test[example,-1,0]) / (np.max(X_test[example,:,0]) - np.min(X_test[example,:,0]))
    X_test[example,:,1] = (X_test[example,:,1] - X_test[example,-1,1]) / (np.max(X_test[example,:,1]) - np.min(X_test[example,:,1]))
    X_test[example,:,2] = (X_test[example,:,2] - X_test[example,-1,2]) / (np.max(X_test[example,:,2]) - np.min(X_test[example,:,2]))
    X_test[example,:,3] = (X_test[example,:,3] - X_test[example,-1,3]) / (np.max(X_test[example,:,3]) - np.min(X_test[example,:,3]))

n_time_features = X_train.shape[1]
classes = Y_train.shape[1]
m_test = X_test.shape[0]
m_train = X_train.shape[0]

X_train = X_train.astype(dtype=np.float32)
X_test = X_test.astype(dtype=np.float32)

############ DEFINE MODEL

# layers = [1, win_size, 100, 1]


data_dim = 4
timesteps = win_size
num_classes = 3


def build_model():
    # expected input data shape: (batch_size, timesteps, data_dim)
    model = Sequential()

    model.add(
        LSTM(
            90,
            return_sequences=True,
            input_shape=(timesteps, data_dim),
            dropout=0.12
        )
    )  # returns a sequence of vectors of dimension 32

    model.add(LSTM(64, return_sequences=True, dropout=0.12))

    model.add(LSTM(32, return_sequences=True, dropout=0.12))

    model.add(LSTM(20, dropout=0.12))  # return a single vector of dimension 32

    model.add(Dense(num_classes, activation='softmax'))

    optimizer = adagrad(lr=0.001)

    start = time.time()
    model.compile(
        loss='categorical_crossentropy',
        optimizer=optimizer,
        metrics=['accuracy']
    )
    print("> Compilation Time : ", time.time() - start)
    return model


def predict_point_by_point(model, data):
    # Predict each timestep given the last sequence of true data, in effect only predicting 1 step ahead each time
    predicted = model.predict(data)
    # predicted = np.reshape(predicted, (predicted.size,))
    return predicted


def predict_sequence_full(model, data, window_size):
    # Shift the window by 1 new prediction each time, re-run predictions on new window
    curr_frame = data[0]
    predicted = []
    for i in range(len(data)):
        predicted.append(model.predict(curr_frame[newaxis, :, :])[0, 0])
        curr_frame = curr_frame[1:]
        curr_frame = np.insert(curr_frame, [window_size - 1], predicted[-1], axis=0)
    return predicted


def predict_sequences_multiple(model, data, window_size, prediction_len):
    # Predict sequence of 50 steps before shifting prediction run forward by 50 steps
    prediction_seqs = []
    for i in range(int(len(data) / prediction_len)):
        curr_frame = data[i * prediction_len]
        predicted = []
        for j in range(prediction_len):
            predicted.append(model.predict(curr_frame[newaxis, :, :])[0, 0])
            curr_frame = curr_frame[1:]
            curr_frame = np.insert(curr_frame, [window_size - 1], predicted[-1], axis=0)
        prediction_seqs.append(predicted)
    return prediction_seqs

######## run model

epochs  = 100  # 100

model = build_model()

csv_logger = CSVLogger('log.csv', append=True, separator=';')
history = model.fit(
    X_train,
    Y_train,
    batch_size=7000,
    epochs=epochs,
    validation_split=0.15,
    callbacks=[csv_logger]
)

model.save('lstm_model_2.h5')
history.save('history.save')

