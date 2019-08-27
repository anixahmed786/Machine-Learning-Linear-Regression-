
# coding: utf-8

# In[9]:

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model


# In[12]:

df= pd.read_csv("E:\Python Data Science\MachineL\homeprices.csv")
df


# In[18]:

get_ipython().magic('matplotlib inline')
plt.xlabel('area(sqr)')
plt.ylabel('price(US)')
plt.scatter(df.area,df.price,color='red',marker='*')


# In[19]:

reg = linear_model.LinearRegression()
reg.fit(df[['area']],df.price)


# In[26]:

reg.predict(3300)


# In[28]:

reg.coef_


# In[31]:

reg.intercept_


# In[33]:

#we know y = mx=c,where m=cof,c=intercept and x=3300

prof = 135.78767123*3300+180616.43835616432
prof


# In[ ]:

#so y=predict

