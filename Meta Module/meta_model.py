#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt


# In[2]:


data=pd.read_csv("Surgical-deepnet.csv")


# In[3]:


data.isnull().sum()


# In[4]:


data.dtypes


# In[5]:


data['gender'].value_counts()


# In[6]:


data.head()


# In[7]:


data.describe()


# In[12]:


plt.plot(data.mean())


# In[11]:


plt.plot(data.var())


# In[8]:


plt.plot(data.skew())


# In[9]:


plt.plot(data.kurtosis())


# In[ ]:




