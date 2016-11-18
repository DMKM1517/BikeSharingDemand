
# coding: utf-8

# In[1]:

from sklearn.neural_network import MLPClassifier
import pandas as pd
import numpy as np
#import matplotlib.pyplot as plt
#get_ipython().magic('matplotlib inline')
from sklearn import preprocessing  
import time


# In[2]:

def rmsle(y_pred, y_actual):
    diff = np.log(y_pred + 1) - np.log(y_actual + 1)
    mean_error = np.square(diff).mean()
    return np.sqrt(mean_error)


# In[3]:

df_train = pd.read_csv("../train.csv")
df_test = pd.read_csv("../test.csv")


# In[4]:

df_train['month'] = pd.DatetimeIndex(df_train.datetime).month
df_train['day'] = pd.DatetimeIndex(df_train.datetime).dayofweek
df_train['hour'] = pd.DatetimeIndex(df_train.datetime).hour
#df_train['dayofm'] = pd.DatetimeIndex(df_train.datetime).day
df_train = df_train.drop(['datetime','casual','registered'],axis=1)


# In[5]:

df_train_train = df_train.drop(['count','holiday','workingday','weather','season'],axis = 1)
df_train_target = df_train['count']


# In[6]:

df_train_train_1 = df_train_train[0:8000]
df_train_train_2 = df_train_train[8000+1:len(df_train_train)]
df_train_target_1 = df_train_target[0:8000]
df_train_target_2 = df_train_target[8000+1:len(df_train_target)]


# In[7]:

def ann_iter(arch,iterations):
    params = {'activation' :'tanh', 
          'solver':'adam', 
          'hidden_layer_sizes':arch,
          'verbose':True,
          'learning_rate':'adaptive',
          'warm_start':False, 
          'tol':1e-30, 
          'max_iter':iterations,
          'early_stopping':False}
    time_start = time.clock()
    ann = MLPClassifier(**params).fit(preprocessing.scale(df_train_train_1),df_train_target_1)
    time_elapsed = (time.clock() - time_start)
    #print time_elapsed
    #print 
    fit = ann.score(preprocessing.scale(df_train_train_1),df_train_target_1)
    pred_test = ann.predict(preprocessing.scale(df_train_train_2))
    pred = np.array(pred_test)
    target = np.array(df_train_target_2.values)
    error =  rmsle(pred_test,target)
    #fitplot = plt.plot(pred_test,target,'.')
    return {'time' : time_elapsed,'fit' : fit, 'error' : error}


# In[8]:

opt_test = pd.DataFrame({'arch': map(lambda p : (100,)*p ,range(100))[1:]})


# In[ ]:

for elem in opt_test['arch']:
    #for elem in fam:
    print (elem, ann_iter(elem,100000))
    #plt.show()


# In[32]:

df_test_test = df_test.drop(['holiday','workingday','weather','season'],axis = 1)
df_test_test['month'] = pd.DatetimeIndex(df_test_test.datetime).month
df_test_test['day'] = pd.DatetimeIndex(df_test_test.datetime).dayofweek
df_test_test['hour'] = pd.DatetimeIndex(df_test_test.datetime).hour
df_test_test = df_test_test.drop(['datetime'],axis = 1)


# In[314]:

#out = ann.predict(df_test_test)


# In[315]:

#pred = pd.DataFrame({'datetime': df_test['datetime'],'count': out})
#pred = pred[['datetime','count']]


# In[316]:

#pred.to_csv("pred.csv", index = False)


# In[ ]:



