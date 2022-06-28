#!/usr/bin/env python
# coding: utf-8

# In[3]:


diabetes


# In[1]:


from sklearn import datasets


# In[2]:


diabetes = datasets.load_diabetes()


# In[4]:


print(diabetes.DESCR)


# In[6]:


print(diabetes.feature_names)
#print features name once again


# In[7]:


X = diabetes.data
Y = diabetes.target


# In[8]:


X.shape, Y.shape


# In[9]:


from sklearn.model_selection import train_test_split
#import necessary library for data split


# In[10]:


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
#data split 20% goes to the test set


# In[11]:


X_train.shape, Y_train.shape
#data dimension i.e 80% training data


# In[13]:


from sklearn import linear_model
#import library for model
from sklearn.metrics import mean_squared_error, r2_score
#import library for computing model


# In[15]:


model = linear_model.LinearRegression()
#Defines the regression model


# In[17]:


model.fit(X_train, Y_train)
# Build actual training model


# In[19]:


Y_pred = model.predict(X_test)
# Apply trained model to make prediction (on test set)


# In[21]:


# **Print model performance**
print('Coefficients:', model.coef_)
#coeffiecient is stored in model.coef
print('Intercept:', model.intercept_)
#intercept stored in model.intercept
print('Mean squared error (MSE): %.2f'
       % mean_squared_error(Y_test, Y_pred))
#mean square stored in mean_squared_error, parameter passed
#Y_test is the actual value, Y_pred is the predicted value.
print('Coefficient of determination (R^2): %.2f'
      % r2_score(Y_test, Y_pred))


# In[22]:


#equation of linear regression model is y=-14.26812489(age) -294.66323133 (sex)+......+ 152.50905379010226
## **String formatting**
r2_score(Y_test, Y_pred)


# In[23]:


r2_score(Y_test, Y_pred).dtype


# In[24]:


'%.2f' %0.49021677588018786


# In[25]:


#Now make scatterplot
# **Import library**
import seaborn as sns


# In[26]:


Y_test
#look at data


# In[27]:


Y_pred


# In[28]:


sns.scatterplot(Y_test, Y_pred)


# In[30]:


sns.scatterplot(Y_test, Y_pred, marker="+")


# In[34]:


sns.scatterplot(Y_test, Y_pred, alpha=0.5)


# In[ ]:




