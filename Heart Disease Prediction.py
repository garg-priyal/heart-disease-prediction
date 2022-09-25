#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# In[6]:


heart_data = pd.read_csv('C:/Users/admin/Downloads/heart_disease_data.csv')


# In[7]:


heart_data.head()


# In[9]:


heart_data.shape


# In[11]:


heart_data.info()


# In[13]:


heart_data.isnull().sum()


# In[14]:


heart_data.describe()


# In[16]:


heart_data['target'].value_counts()


# In[17]:


#1 defectve
#0 healthy
x=heart_data.drop(columns='target',axis=1)
y=heart_data['target']


# In[18]:


print(x)


# In[19]:


print(y)


# In[22]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,stratify=y,random_state=2)


# In[23]:


print(x.shape,x_train.shape,x_test.shape)


# In[24]:


model=LogisticRegression()


# In[25]:


model.fit(x_train,y_train)


# In[26]:


#evaluation of accuracy score on training data
x_train_prediction=model.predict(x_train)
training_data_accuracy=accuracy_score(x_train_prediction, y_train)


# In[29]:


print('Accuracy:', training_data_accuracy)


# In[30]:


x_test_prediction=model.predict(x_test)
test_data_accuracy=accuracy_score(x_test_prediction, y_test)


# In[31]:


print('Accuracy:', test_data_accuracy)


# In[36]:


input_data=(41,0,1,130,204,0,0,172,0,1.4,2,0,2)
data= np.asarray(input_data)
input_data_reshaped=data.reshape(1,-1)
prediction=model.predict(input_data_reshaped)
print(prediction)

if (prediction==0):
    print('The person does not have a heart disease')
else:
    print('The person has heart disease')


# In[ ]:




