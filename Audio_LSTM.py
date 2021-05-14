#!/usr/bin/env python
# coding: utf-8

# In[59]:


import librosa
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# get_ipython().run_line_magic('matplotlib', 'inline')
import os
from PIL import Image
import pathlib
import csv

# Preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

#Keras
import keras

import warnings
warnings.filterwarnings('ignore')


# In[60]:


data = pd.read_csv('./data/Emotion_feature_mood.csv')
data.head()


# Dropping unneccesary columns
data1 = data.drop(['Unnamed: 0'],axis=1)
data2 = data1.drop(['song_name'],axis=1)
data2.head()


# In[63]:


Arousal = data2.iloc[:, -2]
valence = data2.iloc[:, -1]

Y = Arousal
scaler = StandardScaler()
X = scaler.fit_transform(np.array(data2.iloc[:, :-2], dtype = float))


# In[64]:


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)


# In[65]:


X_test = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))
X_train = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))


# In[66]:


print("Training X shape: " + str(X_train.shape))
print("Training Y shape: " + str(Y_train.shape))

print("Test X shape: " + str(X_test.shape))
print("Test Y shape: " + str(Y_test.shape))


# In[67]:



input_shape =(X_train.shape[1],X_train.shape[2])
input_shape


# In[111]:


from keras.models import Sequential
from keras.layers.recurrent import LSTM
from keras.layers import Dense
from keras.optimizers import Adam
from sklearn.metrics import r2_score

print("Build LSTM RNN model ...")

model = Sequential()
# model.add(Dense(256, activation='relu', input_shape=(X_train.shape[1],)))
model.add(LSTM(units=32, dropout=0.05, recurrent_dropout=0.5, return_sequences=True, input_shape=input_shape))
# model.add（Bidirectional(tf.keras.layers.LSTM(32))）

# model.add(LSTM(units=32,  dropout=0.05, recurrent_dropout=0.35, return_sequences=False))
# model.add(Dense(units=Y_train.shape[1], activation="softmax"))
# model.add(Dense(256, activation='relu', input_shape=(X_train.shape[1],)))
model.add(Dense(128, activation='relu'))
model.add(Dense(1, activation='relu'))

# model.add(layers.Dense(64, activation='relu'))

# model.add(layers.Dense(10, activation='softmax'))


# In[112]:

import keras.backend as K

def r2(y_true, y_pred):
    a = K.square(y_pred - y_true)
    b = K.sum(a)
    c = K.mean(y_true)
    d = K.square(y_true - c)
    e = K.sum(d)
    f = 1 - b/e
    return f


print("Compiling ...")
# Keras optimizer defaults:
# Adam   : lr=0.001, beta_1=0.9,  beta_2=0.999, epsilon=1e-8, decay=0.
# RMSprop: lr=0.001, rho=0.9,                   epsilon=1e-8, decay=0.
# SGD    : lr=0.01,  momentum=0.,                             decay=0.
opt = Adam()
# train_score = r2_score(train_label, train_pred)
model.compile(loss="mean_squared_error", optimizer='Adam', metrics=[r2])
model.summary()


# In[113]:




# In[114]:


keras.callbacks.EarlyStopping(monitor='loss', patience=10, verbose=0, mode='min')


# In[133]:


print("Training ...")
batch_size = 32  # num of training examples per minibatch
num_epochs = 1024


model.fit(
    X_train,
    Y_train,
    batch_size=batch_size,
    epochs=num_epochs,)


# In[134]:


print("\nTesting ...")
score, accuracy = model.evaluate(
    X_test, Y_test, batch_size=148, verbose=1
)
print("Test loss:  ", score)
print("Test accuracy:  ", accuracy)


# In[123]:


model_filename = "lstm_genre_classifier_lstm.h5"
print("\nSaving model: " + model_filename)
model.save(model_filename)


# In[ ]:




