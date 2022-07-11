#!/usr/bin/env python
# coding: utf-8

# In[1]:


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

# In[56]:


iris_o = acquire.get_iris_data()


# In[57]:


def prep_iris(df):
    df.drop(columns = ['species_id', 'measurement_id', 'Unnamed: 0'], inplace = True)
    df.rename(columns={"species_name": "species"}, inplace = True)
    dummy_df = pd.get_dummies(df['species'], dummy_na= False)
    df = pd.concat([df, dummy_df], axis=1)
    return df


# In[58]:


def split_iris_data(df):
    '''
    This function performs split on telco data, stratify churn.
    Returns train, validate, and test dfs.
    '''
    train_validate, test = train_test_split(df, test_size=.2, 
                                        random_state=123, 
                                        stratify=df.species)
    train, validate = train_test_split(train_validate, test_size=.2, 
                                   random_state=123, 
                                   stratify=train_validate.species)
    return train, validate, test


# In[59]:


iris = prep_iris(iris_o)


# In[60]:


train, validate, test = split_iris_data(iris)


# ### Titanic

# In[47]:


df = acquire.get_titanic_data()


# In[48]:


def impute_mode(df):
    '''
replace non-existant values before breaking it down into training sets
    '''
    imputer = SimpleImputer(missing_values = np.nan, strategy='most_frequent')
    df[['embark_town']] = imputer.fit_transform(df[['embark_town']])
    return df


# In[49]:


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


# In[50]:


def prep_titanic(df):
    df.drop(columns = ['Unnamed: 0', 'passenger_id', 'deck', 'embarked'], inplace = True)
    dummy_df = pd.get_dummies(df[['sex', 'embark_town', 'class']], dummy_na=False, drop_first= True)
    df = pd.concat([df, dummy_df], axis=1)
    df = impute_mode(df)
    return df
    


# In[51]:


titanic = prep_titanic(df)


# In[52]:


train, validate, test = split_titanic_data(titanic)


# In[15]:


titanic.shape


# In[16]:


train.shape


# In[17]:


test.shape


# In[18]:


validate.shape


# In[19]:


#learnign imputer stuff


# In[20]:


#imputer = SimpleImputer(strategy='most_frequent')


# In[21]:


#imputer = imputer.fit(train[['embark_town']])


# In[22]:


#df[['embark_town']] = imputer.transform(df[['embark_town']])


# In[ ]:





# # Telco

# In[23]:


df = acquire.get_telco_data()


# In[24]:


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


# In[25]:


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


# In[ ]:





# In[46]:


train, validate, test = split_telco_data(df)


# In[26]:


telco = prep_telco(df)


# In[27]:


telco.shape


# In[28]:


train.shape


# In[29]:


validate.shape


# In[30]:


test.shape


# In[ ]:





# In[33]:


#indivudual steps


# In[34]:


df.loc[df.total_charges == ' ', 'total_charges'] = df.monthly_charges


# In[35]:


df.total_charges = df.total_charges.astype('float')


# In[36]:


df.total_charges.sort_values()


# In[37]:


# several of the accounts have no totals


# In[38]:


df[df.total_charges == ' '];


# In[39]:


#it appears that if they are new cosutomers, they dont have totals till after they pay


# In[40]:


df[df.total_charges == ' '];


# In[41]:


df.loc[df.total_charges == ' ', 'total_charges'] = df.monthly_charges


# In[42]:


df[df.tenure == 0];


# In[43]:


df.total_charges = df.total_charges.astype('float')


# In[44]:


df = df[df.total_charges != ' ']


# In[45]:


# def telco_clean_monthly_total (df):
#     if df.tenure == 0:
#         df.total_charges = df.monthly_charges
#     return df

# telco_clean_monthly_total(df)


# In[ ]:





# In[ ]:




