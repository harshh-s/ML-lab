#!/usr/bin/env python
# coding: utf-8

# *Harsh Seksaria*
# 
# *2048011*
# 
# *1-MDS*
# 
# **Date** : *24-01-2021*
# 
# **Topic** : *Exploratory Data Analysis*

# The following dataset contains data of diabetic patients recording different measures of biological factors which help in predicting whether or not the patient is prone to be affected from diabetes.
# 
# The dataset has been downloaded from Kaggle.com

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


fl = pd.read_csv('diabetes.csv')
fl.sample(3)


# In[3]:


fl.describe()


# *The above output generated descriptive statistics summarizing data disribution on numerical values.*

# In[4]:


fl.info(verbose=True)


# *The above output gives us the information about the data types, non-null count, total columns, total records and memory usage.*

# # Data Distribution

# In[6]:


p = fl.hist(figsize=(20,20))


# # Missing Values Imputation

# Wherever we have values as 0, it doesn't make any sense will cause problems in accurately understanding data. They are just like null. So, we first convert them to *null* and then do the missing value imputation.

# In[7]:


fl[['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']] = fl[['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']].replace(0, np.nan)
fl.head(5)


# In[8]:


fl.isnull().sum()


# We can clearly see that there were many 0 in our dataset, and now they have been converted to 0.
# 
# Now, we will fill these null values with appropriate methods to represent the best of data.

# In[9]:


fl['Glucose'].fillna(fl['Glucose'].mean(), inplace=True)
fl['BloodPressure'].fillna(fl['BloodPressure'].mean(), inplace=True)
fl['SkinThickness'].fillna(fl['SkinThickness'].median(), inplace=True)
fl['Insulin'].fillna(fl['Insulin'].median(), inplace=True)
fl['BMI'].fillna(fl['BMI'].median(), inplace=True)


# In[10]:


fl.isnull().sum()


# We first imputed null values by using mean/median whichever was suitable and then checked for null values to verify that there existed none.
# 
# Let's see the histogram again.

# In[12]:


p = fl.hist(figsize=(20,20))


# In[13]:


fl['Outcome'].value_counts().plot(kind='bar', y=['0', '1'])
var = fl['Outcome'].value_counts().to_dict()
print('Diabetic patients: ', str(var[1.0]))
print('Non-Diabetic patients: ', str(var[0.0]))


# We can see that there are 500 patients who don't have diabetes but 268 do suffer from it.

# # Scatter Plot

# In[14]:


p=sns.pairplot(fl, hue = 'Outcome')


# The above graphs show the relationship between two quantities - the measure of the strength of association between two variables.

# # Heat-map
# 
# 
# A heat-map represents the information with the help of different colours, rather, different shades of a color.

# In[15]:


plt.figure(figsize=(12,10))
p=sns.heatmap(fl.corr(), annot=True, cmap ='RdYlGn')


# # Violin Plot
# 
# A violin plot is a method of plotting numeric data. It is similar to box plot with a rotated kernel density plot on each side. Violin plots are similar to box plots, except that they also show the probability density of the data at different values (in the simplest case this could be a histogram).

# In[16]:


df=fl


# In[17]:


fig,ax = plt.subplots(nrows=4, ncols=2, figsize=(18,18))
plt.suptitle('Violin Plots',fontsize=24)
sns.violinplot(x="Pregnancies", data=df,ax=ax[0,0],palette='Set3')
sns.violinplot(x="Glucose", data=df,ax=ax[0,1],palette='Set3')
sns.violinplot (x ='BloodPressure', data=df, ax=ax[1,0], palette='Set3')
sns.violinplot(x='SkinThickness', data=df, ax=ax[1,1],palette='Set3')
sns.violinplot(x='Insulin', data=df, ax=ax[2,0], palette='Set3')
sns.violinplot(x='BMI', data=df, ax=ax[2,1],palette='Set3')
sns.violinplot(x='DiabetesPedigreeFunction', data=df, ax=ax[3,0],palette='Set3')
sns.violinplot(x='Age', data=df, ax=ax[3,1],palette='Set3')
plt.show()


# In[18]:


plt.subplots(figsize=(20,15))
sns.boxplot(x='Outcome', y='Age', data=fl)


# # Conclusion
# 
# We saw the data with the help of different types of graphs which told us how the different variables behave under the influence of other variables.

# ## Supervised Learning
# 
# Supervised learning is the machine learning task of learning a function that maps an input to an output based on example input-output pairs. It infers a function from labeled training data consisting of a set of training examples.
# 
# The most widely used supervised learning algorithms are:
# 1. Support Vector Machines
# 2. Linear Regression
# 3. Logistic Regression
# 4. k-nearest neighbour
# 5. Decision trees
# 6. Naive Bayes
# 7. Linear Discriminant Analysis
# 
# The best suited algorithm for the model is the one that yeilds highest accuracy. Upon implementing and checking the algorithms, we can know which is most accurate and hence choose that one.
# 
# Now, for the dataset selected, supervised learning will be a suitable choice as it contains labels.
