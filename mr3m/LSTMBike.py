
# coding: utf-8

# In[1]:

#%matplotlib inline


# In[2]:

import numpy as np
import numpy
#import matplotlib.pyplot as plt
import pandas as pd
import math
import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, TimeDistributed
from keras.layers import LSTM, SimpleRNN
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import scipy as scipy


# In[3]:

numpy.random.seed(7)


# In[4]:

df_train = pd.read_csv("../train.csv")
df_test = pd.read_csv("../test.csv")


# In[5]:

df_train.head(2)


# In[6]:

df_train_train = df_train.drop(['datetime','casual','registered','count'], axis = 1)
df_test_train = df_train['count']


# In[7]:

n = len(df_train_train)
train_x = df_train_train[0:int(n*.8)].astype('float32')
train_y = df_test_train[0:int(n*.8)].astype('float32')
test_x = df_train_train[int(n*.8)+1:n].astype('float32')
test_y = df_test_train[int(n*.8)+1:n].astype('float32')


# In[8]:

scaler = MinMaxScaler(feature_range=(0, 1))


# In[9]:

train_x_r = scaler.fit_transform(train_x.values)
trainY = scaler.fit_transform(train_y.values)
test_x_r = scaler.fit_transform(test_x.values)
testY = scaler.fit_transform(test_y.values)


# In[10]:

#simplernn = keras.layers.recurrent.SimpleRNN(64, init='glorot_uniform', inner_init='orthogonal', activation='sigmoid', W_regularizer=None, U_regularizer=None, b_regularizer=None, dropout_W=0.0, dropout_U=0.0)


# In[11]:

trainX = numpy.reshape(train_x_r, (train_x_r.shape[0], 1, train_x_r.shape[1]))
testX = numpy.reshape(test_x_r, (test_x_r.shape[0], 1, test_x_r.shape[1]))


# In[12]:

#model = Sequential()
#model.add(Dense(1, input_shape=(8,)))
#model.add(SimpleRNN(64, input_shape=(8,)))
#model.add(Dense(8))
#model.add(Activation('relu'))

# create model
#model = Sequential()
#model.add(Dense(13, input_dim=8, init='normal', activation='relu'))
#model.add(Dense(1, init='normal'))
# Compile model
#model.compile(loss='mean_squared_error', optimizer='adam')

#model = Sequential()
#model.add(Dense(1, input_shape=(8,) ))
#model.add(Activation('relu'))
#model.add(LSTM(2, input_shape=(1, 8)))


#model.compile(loss='mean_squared_error', optimizer='adam')
#model.fit(trainX, trainY, nb_epoch=10, batch_size=32,verbose=1)


# In[13]:

from keras.models import Sequential
from keras.layers import LSTM, Dense
import numpy as np

data_dim = 8
timesteps = 1

# expected input data shape: (batch_size, timesteps, data_dim)
model = Sequential()
model.add(LSTM(32, return_sequences=True,
               input_shape=(timesteps, data_dim)))  # returns a sequence of vectors of dimension 32
model.add(LSTM(32, return_sequences=True))  # returns a sequence of vectors of dimension 32
model.add(LSTM(32))  # return a single vector of dimension 32
model.add(Dense(1, activation='linear'))

model.compile(loss='mean_squared_error',
              optimizer='rmsprop')

# generate dummy training data
x_train = numpy.reshape(train_x_r, (train_x_r.shape[0], timesteps, train_x_r.shape[1]))
y_train = scaler.fit_transform(train_y.values)


x_val = numpy.reshape(test_x_r, (test_x_r.shape[0], timesteps, test_x_r.shape[1]))
y_val = scaler.fit_transform(test_y.values)
# generate dummy validation data

model.fit(x_train, y_train,
          batch_size=64, nb_epoch=100,
          validation_data=(x_val, y_val))


# In[14]:

trainPredict = model.predict(x_train)
testPredict = model.predict(x_val)
trainPredict = scaler.inverse_transform(trainPredict)
train_y_r_e = scaler.inverse_transform([y_train])
testPredict = scaler.inverse_transform(testPredict)
test_y_r_e = scaler.inverse_transform([y_val])


# In[15]:

trainScore = math.sqrt(mean_squared_error(train_y_r_e[0], trainPredict))
print('Train Score: %.2f RMSE' % (trainScore))
testScore = math.sqrt(mean_squared_error(test_y_r_e[0], testPredict))
print('Test Score: %.2f RMSE' % (testScore))


# In[16]:

print scipy.stats.pearsonr(train_y_r_e[0],trainPredict[:,0])[0]
#plt.plot(train_y_r_e[0],trainPredict,'.')


# In[17]:

print scipy.stats.pearsonr(test_y_r_e[0],testPredict[:,0])[0]
#plt.plot(test_y_r_e[0],testPredict,'.')


# In[18]:

df_test_test = df_test.drop(['datetime'], axis = 1)
df_test_t = scaler.fit_transform(df_test_test.values)
df_test_r = numpy.reshape(df_test_t, (df_test_t.shape[0], timesteps , df_test_t.shape[1]))
pred = model.predict(df_test_r)


# In[21]:

scaler = MinMaxScaler(feature_range=(0, 1))
scaler.fit_transform(test_y.values)


# In[22]:

out = scaler.inverse_transform(pred).astype(int)
out[out<0] = 0
out = out[:,0]


# In[23]:

pred = pd.DataFrame({'datetime': df_test['datetime'],'count': out})
pred = pred[['datetime','count']]


# In[24]:

pred.to_csv("pred.csv", index = False)


# In[ ]:



