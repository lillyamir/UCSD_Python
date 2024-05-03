#!/usr/bin/env python
# coding: utf-8

# In[7]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


# In[8]:


data = pd.read_csv('World Energy Consumption.csv')


# In[9]:


#Creating new df
pd.options.mode.chained_assignment = None 
recent = data['year']==2015



recent_year = data[recent]
recent_year



#New df columns
recent_year['gdp_percap'] = recent_year['gdp']/recent_year['population'] #Creating a new column for our dependent variable.
recent_year = recent_year.dropna(subset=['gdp_percap', 'year', 'fossil_energy_per_capita']) #Dropping rows in the gdp_percap column that are N/A values.


recent_year


# In[10]:


x_vals = np.array(recent_year['gdp_percap'])
x_vals = x_vals.reshape(len(x_vals), 1)
y_vals = recent_year[ 'fossil_energy_per_capita']


# In[11]:



xtrain, xtest, ytrain, ytest = train_test_split(x_vals, y_vals, test_size = 0.25)
# Initialize the model
mod_tr = LinearRegression()
# Fit the model
mod_tr = mod_tr.fit(X = xtrain, y = ytrain)

mod_tr.intercept_
mod_tr.coef_

mod_preds = mod_tr.predict(X = xtest)

mod_preds


sns.scatterplot(x = xtest.reshape(len(xtest)),
                y = ytest
               )

plt.axline(xy1 = (0, mod_tr.intercept_), slope = mod_tr.coef_[0], color = "r", linestyle = "--")

plt.title("Regression performance on held out test data")


# In[29]:


# Training MSE

# Generate model predictions for the training data
train_pred = mod_tr.predict(X = xtrain)
# Calculate the squared error of these predictions
train_error = (ytrain - train_pred)**2
# Get the mean squared error
train_mse = train_error.sum() / (len(train_error) - 2)


# Test MSE

# Generate model predictions for the test data (we did this above already)
test_pred = mod_tr.predict(X = xtest)
# Calculate the squared error of the test predictions
test_error = (ytest - test_pred)**2
# Get the mean squared error for test data
test_mse = test_error.sum() / (len(test_error) - 2)
print("train mse", test_mse, 'test_mse', train_mse)


# In[15]:


mod_tr.score(X = x_vals, y = y_vals)


# In[16]:


from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters = 3, # default value is 8
                random_state = 1)

kmeans.fit(X = x_vals)

kmeans


# In[17]:


#Cluster Centers
kmeans.cluster_centers_

y_kmeans = kmeans.predict(X = x_vals)
len(y_kmeans)


# In[18]:


kmeans.labels_

g = sns.scatterplot(data = recent_year, x = "gdp_percap", y = "fossil_energy_per_capita", hue = kmeans.labels_)
g.set_xscale('log')


# In[19]:


recent_cluster_map = pd.DataFrame()
recent_cluster_map['country'] = recent_year['country']
recent_cluster_map['cluster'] = kmeans.labels_
recent_cluster_map


# In[20]:


pd.options.mode.chained_assignment = None 
old = data['year']==1985



old_year = data[old]
old_year



#New df columns
old_year['gdp_percap'] = old_year['gdp']/old_year['population'] #Creating a new column for our dependent variable.
old_year = old_year.dropna(subset=['gdp_percap', 'year', 'fossil_energy_per_capita']) #Dropping rows in the gdp_percap column that are N/A values.


old_year

x_vals2 = np.array(old_year['gdp_percap'])
x_vals2 = x_vals2.reshape(len(x_vals2), 1)
y_vals2 = old_year[ 'fossil_energy_per_capita']


# In[21]:


from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters = 3, # default value is 8
                random_state = 1)

kmeans.fit(X = x_vals2)

kmeans


# In[22]:


#Cluster Centers
kmeans.cluster_centers_

y_kmeans = kmeans.predict(X = x_vals2)
len(y_kmeans)


# In[23]:


kmeans.labels_

g = sns.scatterplot(data = old_year, x = "gdp_percap", y = "fossil_energy_per_capita", hue = kmeans.labels_)
g.set_xscale('log')


# In[24]:


old_cluster_map = pd.DataFrame()
old_cluster_map['country'] = old_year['country']
old_cluster_map['cluster'] = kmeans.labels_
old_cluster_map


# In[ ]:




