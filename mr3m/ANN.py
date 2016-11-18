
# coding: utf-8

# In[1]:

from sklearn.neural_network import MLPRegressor
import pandas as pd
import numpy as np
#import matplotlib.pyplot as plt
#get_ipython().magic('matplotlib inline')
from sklearn import preprocessing  
import time


# In[2]:

def rmsle(y_pred, y_actual):
    y_pred.astype(int)
    y_pred[y_pred<0] = 0
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
#df_train = df_train.drop(['datetime','casual','registered'],axis=1)
df_train = df_train.drop(['datetime'],axis=1)


# In[5]:

df_train_train = df_train.drop(['count','registered','casual','holiday','workingday','weather','season'],axis = 1)
#df_train_train = df_train.drop(['count','registered','casual'],axis = 1)
df_train_target_a = df_train['casual']
df_train_target_b = df_train['registered']
df_train_target_c = df_train['count']


# In[6]:

def ann_iter(df_in,df_target,arch,iterations):
    params = {'activation' :'tanh', 
          'solver':'adam', 
          'hidden_layer_sizes': arch,
          'verbose':True,
          'learning_rate':'adaptive',
          'warm_start':False, 
          'tol':1e-30, 
          'max_iter':iterations,
          'early_stopping':False}
    df_train_train_a_1 = df_in[0:int(len(df_in)*.8)]
    df_train_train_a_2 = df_in[int(len(df_in)*.8)+1:len(df_in)]
    df_train_target_a_1 = df_target[0:int(len(df_in)*.8)]
    df_train_target_a_2 = df_target[int(len(df_in)*.8)+1:len(df_in)]
    time_start = time.clock()
    ann = MLPRegressor(**params).fit(df_train_train_a_1,df_train_target_a_1)
    time_elapsed = (time.clock() - time_start)
    #print time_elapsed
    #print 
    fit = ann.score(df_train_train_a_1,df_train_target_a_1)
    pred_test = ann.predict(df_train_train_a_2)
    pred = np.array(pred_test)
    target = np.array(df_train_target_a_2.values)
    error =  rmsle(pred_test,target)
   # fitplot = plt.plot(pred_test,target,'.')
    return {'time' : time_elapsed,'fit' : fit, 'error' : error, 'ann':ann}


# In[7]:

#opt_test = pd.DataFrame({'arch': map(lambda p : (10,)*p ,range(100))[1:]})


# In[8]:

#for fam in opt_test['arch']:
#    for elem in fam:
#        print elem, ann_iter(elem,1000)
#    plt.show()


# In[9]:

arch = (100,)*100


# In[10]:

ann1 = ann_iter(df_train_train,df_train_target_a,arch ,10000)
print ann1
ann2 = ann_iter(df_train_train,df_train_target_b,arch ,10000)
print ann2
#ann3 = ann_iter(df_train_train,df_train_target_c,arch ,10000)
#print ann3


# In[ ]:

#df_test_test = df_test
df_test_test = df_test.drop(['holiday','workingday','weather','season'],axis = 1)
df_test_test['month'] = pd.DatetimeIndex(df_test_test.datetime).month
df_test_test['day'] = pd.DatetimeIndex(df_test_test.datetime).dayofweek
df_test_test['hour'] = pd.DatetimeIndex(df_test_test.datetime).hour
df_test_test = df_test_test.drop(['datetime'],axis = 1)


# In[ ]:

out1 = ann1['ann'].predict(df_test_test)
out2 = ann2['ann'].predict(df_test_test)
#out3 = ann3['ann'].predict(df_test_test)
out = out1 + out2


# In[ ]:

out = out.astype(int)
out[out<0] = 0


# In[ ]:

pred = pd.DataFrame({'datetime': df_test['datetime'],'count': out})
pred = pred[['datetime','count']]


# In[ ]:

pred.to_csv("pred3.csv", index = False)


# In[ ]:

#plt.plot(out1,out2,'.')


# In[ ]:

#plt.plot(out,out3,'.')


# In[ ]:

#plt.plot(out,'.')


# In[ ]:
#
#plt.plot(df_train_target_a,ann1['ann'].predict(preprocessing.scale(df_train_train)) ,'.')
#plt.plot(df_train_target_b,ann2['ann'].predict(preprocessing.scale(df_train_train)) ,'.')


# In[ ]:

#plt.plot(ann1['ann'].predict(preprocessing.scale(df_train_train)) ,'.')
#plt.plot(ann2['ann'].predict(preprocessing.scale(df_train_train)) ,'.')


# In[ ]:



