#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import seaborn as sns
import scipy.stats as stats
import pandas as pd
import matplotlib.pyplot as plt
from pydataset import data
import env 
import acquire


# telco = pd.read_csv("telco_churn.csv")

# sleep = data('sleepstudy')

# mpg = data('mpg')

# data(show_doc = True);

# # Data Preparation

# ### IRIS


iris = acquire.get_iris_data()


# In[57]:


def prep_iris(df):
    df.drop(columns = ['species_id', 'measurement_id', 'Unnamed: 0'], inplace = True)
    df.rename(columns={"species_name": "species"}, inplace = True)
    dummy_df = pd.get_dummies(df['species'], dummy_na= False)
    df = pd.concat([df, dummy_df], axis=1)
    return df


# In[58]:


iris = prep_iris(iris)


# ### Titanic

# In[60]:


df = acquire.get_titanic_data()


# In[67]:


def prep_titanic(df):
    df.drop(columns = ['Unnamed: 0', 'passenger_id', 'deck', 'embarked'], inplace = True)
    dummy_df = pd.get_dummies(df[['sex', 'embark_town', 'class']], dummy_na=False, drop_first= True)
    df = pd.concat([df, dummy_df], axis=1)
    return df
    
    


# In[70]:


prep_titanic(df)


# # Telco

# In[71]:


df = acquire.get_telco_data()


# In[78]:


def prep_telco(df):
    df.drop(columns = ['Unnamed: 0', 'payment_type_id', 'internet_service_type_id', 'contract_type_id'], inplace = True)
    dummy_df = pd.get_dummies(df[['contract_type', 'internet_service_type', 'payment_type']], dummy_na=False, drop_first= True)
    df = pd.concat([df, dummy_df], axis=1)
    return df


# In[79]:


prep_telco(df)


# In[ ]:




