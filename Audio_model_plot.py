#!/usr/bin/env python
# coding: utf-8

# In[1]:


import librosa
import pandas as pd
import numpy as np
from IPython import get_ipython
import matplotlib.pyplot
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


# In[3]:


data = pd.read_csv('./data/Emotion_feature_mood.csv')
data.head()


# In[4]:


data.shape


# In[5]:


# Dropping unneccesary columns
data1 = data.drop(['Unnamed: 0'],axis=1)
data2 = data1.drop(['song_name'],axis=1)
data2.head()


# In[6]:


Arousal = data2.iloc[:, -2]
valence = data2.iloc[:, -1]

Y = Arousal
scaler = StandardScaler()
X = scaler.fit_transform(np.array(data2.iloc[:, :-2], dtype = float))


# In[7]:


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)


# In[8]:


X_test = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))
X_train = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))


# In[9]:


print("Training X shape: " + str(X_train.shape))
print("Training Y shape: " + str(Y_train.shape))

print("Test X shape: " + str(X_test.shape))
print("Test Y shape: " + str(Y_test.shape))


# In[10]:



input_shape =(X_train.shape[1],X_train.shape[2])
input_shape


# In[11]:


from keras.models import Sequential
from keras.layers.recurrent import LSTM
from keras.layers import Dense
from keras.optimizers import Adam
from sklearn.metrics import r2_score

print("Build LSTM model ...")

model = Sequential()
# model.add(Dense(256, activation='relu', input_shape=(X_train.shape[1],)))
model.add(LSTM(units=32, dropout=0.05, recurrent_dropout=0.5, return_sequences=True, input_shape=input_shape))
# model.add（Bidirectional(tf.keras.layers.LSTM(32))）

model.add(LSTM(units=16,  dropout=0.05, recurrent_dropout=0.35, return_sequences=False))
# model.add(Dense(units=Y_train.shape[1], activation="softmax"))
# model.add(Dense(256, activation='relu', input_shape=(X_train.shape[1],)))
model.add(Dense(128, activation='relu'))
model.add(Dense(1, activation='relu'))

# model.add(layers.Dense(64, activation='relu'))

# model.add(layers.Dense(10, activation='softmax'))


# In[13]:


import keras.backend as K

def r2(y_true, y_pred):
    a = K.square(y_pred - y_true)
    b = K.sum(a)
    c = K.mean(y_true)
    d = K.square(y_true - c)
    e = K.sum(d)
    f = 1 - b/e
    return f


# In[14]:



print("Compiling ...")
# Keras optimizer defaults:
# Adam   : lr=0.001, beta_1=0.9,  beta_2=0.999, epsilon=1e-8, decay=0.
# RMSprop: lr=0.001, rho=0.9,                   epsilon=1e-8, decay=0.
# SGD    : lr=0.01,  momentum=0.,                             decay=0.
opt = Adam()
# train_score = r2_score(train_label, train_pred)
model.compile(loss="mean_squared_error", optimizer='Adam', metrics=[r2])
model.summary()


# In[15]:


keras.callbacks.EarlyStopping(monitor='loss', patience=10, verbose=0, mode='min')


# In[27]:


print("Training ...")
batch_size = 16  # 32
num_epochs = 2000 #2000


twoLSTM=model.fit(
    X_train,
    Y_train,
    batch_size=batch_size,
    epochs=num_epochs,
)


# In[28]:


print("\nTesting ...")
score, accuracy = model.evaluate(
    X_test, Y_test, batch_size=148, verbose=1
)
print("Test loss:  ", score)
print("Test accuracy:  ", accuracy)


# In[ ]:





# In[22]:


from pandas import DataFrame
from matplotlib import pyplot

au_Loss=DataFrame()
au_R2=DataFrame()


# In[23]:



au_Loss['2LSTM']=twoLSTM.history['loss']
au_R2['2LSTM']=twoLSTM.history['r2']


# In[ ]:





# In[24]:


print("Build LSTM model ...")

LSTMmodel = Sequential()
# model.add(Dense(256, activation='relu', input_shape=(X_train.shape[1],)))
LSTMmodel.add(LSTM(units=32, dropout=0.05, recurrent_dropout=0.5, return_sequences=True, input_shape=input_shape))
# model.add（Bidirectional(tf.keras.layers.LSTM(32))）

# model.add(LSTM(units=16,  dropout=0.05, recurrent_dropout=0.35, return_sequences=False))
# model.add(Dense(units=Y_train.shape[1], activation="softmax"))
# model.add(Dense(256, activation='relu', input_shape=(X_train.shape[1],)))
LSTMmodel.add(Dense(128, activation='relu'))
LSTMmodel.add(Dense(1, activation='relu'))

# model.add(layers.Dense(64, activation='relu'))

# model.add(layers.Dense(10, activation='softmax'))

print("Compiling ...")
# Keras optimizer defaults:
# Adam   : lr=0.001, beta_1=0.9,  beta_2=0.999, epsilon=1e-8, decay=0.
# RMSprop: lr=0.001, rho=0.9,                   epsilon=1e-8, decay=0.
# SGD    : lr=0.01,  momentum=0.,                             decay=0.
opt = Adam()
# train_score = r2_score(train_label, train_pred)
LSTMmodel.compile(loss="mean_squared_error", optimizer='Adam', metrics=[r2])
LSTMmodel.summary()
print("Training ...")
batch_size = 16  # 8
num_epochs = 100 #2000


au_LSTM=LSTMmodel.fit(
    X_train,
    Y_train,
    batch_size=batch_size,
    epochs=num_epochs,
)
print("\nTesting ...")
score, accuracy = LSTMmodel.evaluate(
    X_test, Y_test, batch_size=148, verbose=1
)
print("Test loss:  ", score)
print("Test accuracy:  ", accuracy)


# In[205]:



au_Loss['LSTM']=au_LSTM.history['loss']
au_R2['LSTM']=au_LSTM.history['r2']


# In[206]:


from keras.models import Sequential
from keras.layers.recurrent import LSTM
from keras.layers import Dense
from keras.optimizers import Adam
from sklearn.metrics import r2_score
from keras.layers import Dense, LSTM, Lambda, TimeDistributed, Input, Masking, Bidirectional,GRU
print("Build LSTM model ...")

Bimodel = Sequential()
# model.add(Dense(256, activation='relu', input_shape=(X_train.shape[1],)))
# LSTMmodel.add(LSTM(units=32, dropout=0.05, recurrent_dropout=0.5, return_sequences=True, input_shape=input_shape))
# Bimodel.add(Bidirectional(LSTM(units=32,return_sequences=False)))
Bimodel.add(GRU(32, input_shape=input_shape))

# model.add(LSTM(units=16,  dropout=0.05, recurrent_dropout=0.35, return_sequences=False))
# model.add(Dense(units=Y_train.shape[1], activation="softmax"))
# model.add(Dense(256, activation='relu', input_shape=(X_train.shape[1],)))
Bimodel.add(Dense(128, activation='relu'))
Bimodel.add(Dense(1, activation='relu'))

# model.add(layers.Dense(64, activation='relu'))

# model.add(layers.Dense(10, activation='softmax'))

print("Compiling ...")
# Keras optimizer defaults:
# Adam   : lr=0.001, beta_1=0.9,  beta_2=0.999, epsilon=1e-8, decay=0.
# RMSprop: lr=0.001, rho=0.9,                   epsilon=1e-8, decay=0.
# SGD    : lr=0.01,  momentum=0.,                             decay=0.
opt = Adam()
# train_score = r2_score(train_label, train_pred)
Bimodel.compile(loss="mean_squared_error", optimizer='Adam', metrics=[r2])
Bimodel.summary()
print("Training ...")
batch_size = 16  # num of training examples per minibatch
num_epochs = 100


GRU=Bimodel.fit(
    X_train,
    Y_train,
    batch_size=batch_size,
    epochs=num_epochs,
)
print("\nTesting ...")
score, accuracy = Bimodel.evaluate(
    X_test, Y_test, batch_size=148, verbose=1
)
print("Test loss:  ", score)
print("Test accuracy:  ", accuracy)


# In[209]:




au_Loss['GRU']=GRU.history['loss']
au_R2['GRU']=GRU.history['r2']
print('loss')
au_Loss.plot()
print('R2Score')
au_R2.plot()
au_Loss.plot(title='Audio Training Loss in different methods',fontsize=10)


# In[ ]:





# In[ ]:




