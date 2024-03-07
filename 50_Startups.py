#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Multiple Linear Regression


# In[2]:


# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# In[42]:


# Importing the dataset
dataset = pd.read_csv(R"J:\1.ML_PROJECT\1.ML_REGGRESION_PROJECTS\2.MULTI LEANIAR\50_Startups.csv")


# In[43]:


dataset.head()


# In[ ]:


### first four colums are independent variabels .profit is dependent variable


# In[ ]:


### preprocssing the data will checking the nul points


# In[44]:


dataset.isna().sum()


# In[ ]:


### it describe how many rows and colums are there in the data


# In[45]:


dataset.shape


# In[ ]:


###featuer matrix


# In[56]:


X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values
print(X)


# In[ ]:


# Encoding categorical data


# In[57]:


from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [3])], remainder='passthrough')
X = np.array(ct.fit_transform(X))


# In[ ]:


Encoder converting categorical variables to numerical variables


# In[58]:



print(X)


# In[ ]:


# Splitting the dataset into the Training set and Test set


# In[65]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


# In[66]:


X_train


# In[ ]:


# Training the Multiple Linear Regression model on the Training set


# In[67]:


from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)


# In[ ]:


# Predicting the Test set results


# In[68]:


y_pred = regressor.predict(X_test)


# In[69]:


from sklearn.metrics import r2_score
r2_score(y_test,y_pred)


# In[ ]:


accuracy value is 0.93
93%

