#!/usr/bin/env python
# coding: utf-8

# In[1]:


#importing the required libraries
import numpy as np
import pandas as pd
import seaborn as sns 
from plotnine import *
import matplotlib.pyplot as plt


# In[3]:


#importing the dataset
nba=pd.read_csv("nba_2013.csv")
nba.head(7)


# In[4]:


nba.shape


# In[5]:


nba.mean()


# In[6]:


#average field goals (fg) made for the season
nba.loc[:,"fg"].mean()


# In[7]:


#creating pairwise scatter plots
sns.pairplot(nba[["ast", "fg", "trb"]])
plt.show()


# In[8]:


#heat map of the columns assists (ast), field goals(fg), and total rebounds(trb).
correlation = nba[["ast", "fg", "trb"]].corr()
sns.heatmap(correlation, annot=True)


# In[9]:


#5 clusters of players using the machine learning model called KMeans 
from sklearn.cluster import KMeans
kmeans_model = KMeans(n_clusters=5, random_state=1)
good_columns = nba._get_numeric_data().dropna(axis=1)
kmeans_model.fit(good_columns)
labels = kmeans_model.labels_
labels


# In[10]:


#visualization of this data
#Principal Component Analysis (PCA)
from sklearn.decomposition import PCA
pca_2 = PCA(2)
plot_columns = pca_2.fit_transform(good_columns)
plt.scatter(x=plot_columns[:,0], y=plot_columns[:,1], c=labels)
plt.show()


# In[11]:


#coordinates / points of each player
plot_columns


# In[12]:


#finding LeBron James and Kevin Durant (two great basketball players) 
LeBron = good_columns.loc[ nba['player'] == 'LeBron James',: ]
Durant = good_columns.loc[ nba['player'] == 'Kevin Durant',: ]
print(LeBron)
print(Durant)


# In[14]:


#finding which cluster these two players are in.
Lebron_list = LeBron.values.tolist()
Durant_list = Durant.values.tolist()
#Predicting which group they belong to
LeBron_Cluster_Label = kmeans_model.predict(Lebron_list)
Durant_Cluster_Label = kmeans_model.predict(Durant_list)

print(LeBron_Cluster_Label)
print(Durant_Cluster_Label)


# In[ ]:


# we can see they both belong to cluster 3


# In[15]:


#correlation
nba.corr()


# In[16]:


# we can see a positive correlation between minutes played (mp) and points(pts).


# In[20]:


#prediction the number of point (pts)per player from field goals (fg)made
#split the data into 80% training and 20% testing.
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(nba[['fg']], nba[['pts']], test_size=0.2, random_state=42)


# In[21]:


#using linear regression model to make prediction
from sklearn.linear_model import LinearRegression
lr = LinearRegression() #model creation
lr.fit(x_train, y_train) #model training
predictions = lr.predict(x_test) # predictions on the test data
print(predictions)
print(y_test)


# In[22]:


lr_confidence = lr.score(x_test, y_test)
print("lr confidence (R^2): ", lr_confidence)


from sklearn.metrics import mean_squared_error
print("Mean Squared Error (MSE): ",mean_squared_error(y_test, predictions))


# In[ ]:





# In[ ]:




