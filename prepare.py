#!/usr/bin/env python
# coding: utf-8

# In[44]:


import numpy as np
import seaborn as sns
import scipy.stats as stats
import pandas as pd
import matplotlib.pyplot as plt
from pydataset import data
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
import env 
import acquire


# In[2]:


# pd.set_option('display.max_columns')
# ValueError: Must provide an even number of non-keyword arguments


# In[3]:


pd.set_option('display.max_columns', None)


# telco = pd.read_csv("telco_churn.csv")

# sleep = data('sleepstudy')

# mpg = data('mpg')

# data(show_doc = True);

# # Data Preparation

# ### IRIS
# 
# Using the Iris Data:
# 
#     Use the function defined in acquire.py to load the iris data.
# 
#     Drop the species_id and measurement_id columns.
# 
#     Rename the species_name column to just species.
# 
#     Create dummy variables of the species name and concatenate onto the iris dataframe. (This is for practice, we don't always have to encode the target, but if we used species as a feature, we would need to encode it).
# 
#     Create a function named prep_iris that accepts the untransformed iris data, and returns the data with the transformations above applied.
# 

# In[4]:


iris = acquire.get_iris_data()


# In[5]:


def prep_iris(df):
    df.drop(columns = ['species_id', 'measurement_id', 'Unnamed: 0'], inplace = True)
    df.rename(columns={"species_name": "species"}, inplace = True)
    dummy_df = pd.get_dummies(df['species'], dummy_na= False)
    df = pd.concat([df, dummy_df], axis=1)
    return df


# In[6]:


iris = prep_iris(iris)


# In[7]:


iris.head()


# ### Titanic

# In[143]:


df = acquire.get_titanic_data()


# In[144]:


def impute_mode(df):
    '''
replace non-existant values before breaking it down into training sets
    '''
    imputer = SimpleImputer(missing_values = np.nan, strategy='most_frequent')
    df[['embark_town']] = imputer.fit_transform(df[['embark_town']])
    return df


# In[145]:


def split_titanic_data(df):
    '''
    This function performs split on telco data, stratify churn.
    Returns train, validate, and test dfs.
    '''
    train_validate, test = train_test_split(df, test_size=.2, 
                                        random_state=123, 
                                        stratify=df.survived)
    train, validate = train_test_split(train_validate, test_size=.2, 
                                   random_state=123, 
                                   stratify=train_validate.survived)
    return train, validate, test


# In[146]:


def prep_titanic(df):
    df.drop(columns = ['Unnamed: 0', 'passenger_id', 'deck', 'embarked'], inplace = True)
    dummy_df = pd.get_dummies(df[['sex', 'embark_town', 'class']], dummy_na=False, drop_first= True)
    df = pd.concat([df, dummy_df], axis=1)
    df = impute_mode(df)
    return df

train, validate, test = split_titanic_data(df)
    
    


# In[147]:


titanic = prep_titanic(df)


# In[148]:


titanic.info()


# In[149]:


titanic.shape


# In[150]:


train.shape


# In[151]:


test.shape


# In[152]:


validate.shape


# In[ ]:


#learnign imputer stuff


# In[ ]:


#imputer = SimpleImputer(strategy='most_frequent')


# In[ ]:


#imputer = imputer.fit(train[['embark_town']])


# In[ ]:


#df[['embark_town']] = imputer.transform(df[['embark_town']])


# In[ ]:





# # Telco

# In[71]:


df = acquire.get_telco_data()


# In[72]:


def split_telco_data(df):
    '''
    This function performs split on telco data, stratify churn.
    Returns train, validate, and test dfs.
    '''
    train_validate, test = train_test_split(df, test_size=.2, 
                                        random_state=123, 
                                        stratify=df.churn)
    train, validate = train_test_split(train_validate, test_size=.2, 
                                   random_state=123, 
                                   stratify=train_validate.churn)
    return train, validate, test


# In[73]:


def prep_telco(df):
    df.drop(columns = ['Unnamed: 0', 'payment_type_id', 'internet_service_type_id', 'contract_type_id', 'customer_id'], inplace = True)
    dummy_df = pd.get_dummies(df[['contract_type', 'internet_service_type', 'payment_type']], dummy_na=False, drop_first= True)
    df = pd.concat([df, dummy_df], axis=1)
    df = df[df.total_charges != ' ']
    df.total_charges = df.total_charges.astype(float)
    
    
    # encode binary categorical variables into numeric values
    df['gender_encoded'] = df.gender.map({'Female': 1, 'Male': 0})
    df['partner_encoded'] = df.partner.map({'Yes': 1, 'No': 0})
    df['dependents_encoded'] = df.dependents.map({'Yes': 1, 'No': 0})
    df['phone_service_encoded'] = df.phone_service.map({'Yes': 1, 'No': 0})
    df['paperless_billing_encoded'] = df.paperless_billing.map({'Yes': 1, 'No': 0})
    df['churn_encoded'] = df.churn.map({'Yes': 1, 'No': 0})
    return df

train, validate, test = split_telco_data(df)


# In[75]:


prep_telco(df)


# In[76]:


telco.shape


# In[77]:


train.shape


# In[78]:


validate.shape


# In[79]:


test.shape


# In[ ]:





# In[33]:


telco.head()


# In[14]:


telco.info()


# In[15]:


#indivudual steps


# In[16]:


df.loc[df.total_charges == ' ', 'total_charges'] = df.monthly_charges


# In[17]:


df.total_charges = df.total_charges.astype('float')


# In[18]:


df.total_charges.sort_values()


# In[19]:


# several of the accounts have no totals


# In[20]:


df[df.total_charges == ' '];


# In[21]:


#it appears that if they are new cosutomers, they dont have totals till after they pay


# In[22]:


df[df.total_charges == ' '];


# In[23]:


df.loc[df.total_charges == ' ', 'total_charges'] = df.monthly_charges


# In[24]:


df[df.tenure == 0];


# In[25]:


df.total_charges = df.total_charges.astype('float')


# In[26]:


df = df[df.total_charges != ' ']


# In[27]:


# def telco_clean_monthly_total (df):
#     if df.tenure == 0:
#         df.total_charges = df.monthly_charges
#     return df

# telco_clean_monthly_total(df)


# In[ ]:





# In[ ]:




