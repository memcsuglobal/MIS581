#!/usr/bin/env python
# coding: utf-8

# In[12]:


import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import train_test_split,cross_val_score,GridSearchCV


# In[15]:


df = pd.read_csv('Desktop/CSU/MIS581/Churn_Modelling.csv')


# In[16]:


df.columns


# In[17]:


df.isnull().sum()


# In[18]:


df = df.drop(['CustomerId','RowNumber','Surname'], axis = "columns")
df.head()


# In[19]:


df.describe().T


# In[26]:


df1 = pd.get_dummies(df,columns = ['Geography','Gender'])
df1.head()


# In[48]:


plt.figure(figsize=(15,12))  
sns.heatmap(df.corr(),annot=True,linewidths=.5, cmap="RdYlGn")
plt.show()


# In[34]:


fig, axes = plt.subplots(2,4,figsize=(12,6))
feats = ['CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'HasCrCard', 'IsActiveMember', 'EstimatedSalary']
for i, ax in enumerate(axes.flatten()):
    ax.hist(df[feats[i]], bins=25, color='orange')
    ax.set_title(str(feats[i])+' Histogram', color='black')
    ax.set_yscale('log')
plt.tight_layout()


# In[37]:


CreditScore_Exite_rate = df.groupby('CreditScore').Exited.mean()
CreditScore_Exite_rate


# In[43]:


plt.figure(figsize = (14,7))
plt.scatter(x=CreditScore_Exite_rate.index , y= CreditScore_Exite_rate.values,color = 'orange',marker = 'o',)
plt.xlabel('CreditScore',fontsize = 15)
plt.ylabel('Exite Rate',fontsize = 15)
plt.title('Churn Rate by Credit Score',fontsize = 20)
plt.show()


# In[39]:


Age_Exite_rate = df.groupby('Age').Exited.mean()
Age_Exite_rate


# In[42]:


plt.figure(figsize = (14,7))
plt.scatter(x=Age_Exite_rate.index , y= Age_Exite_rate.values,color = 'orange',marker = 'o',)
plt.xlabel('Age',fontsize = 15)
plt.ylabel('Exite Rate',fontsize = 15)
plt.title('Churn Rate by Age',fontsize = 20)
plt.show()


# In[47]:


labels = 'Churned', 'Retained'
sizes = [df.Exited[df['Exited']==1].count(), df.Exited[df['Exited']==0].count()]
explode = (0, 0.1)
fig1, ax1 = plt.subplots(figsize=(10, 8))
ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',
        shadow=False, startangle=90)
ax1.axis('equal')
plt.show()

