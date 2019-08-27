
# coding: utf-8

# In[54]:

#All imports
get_ipython().magic('matplotlib inline')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#plt.rcParams['figure.figsize']=(20,0,10,0)


# In[56]:

#Reading the data of headbrain.csv file 
data = pd.read_csv('E:\Python Data Science\MYWORK\headbrain.csv')
#print(data.Gender)
data.head()


# In[57]:

#Collect the particular value from the CSV file and assigned to X and Y
X = data['Head Size(cm^3)'].values
Y = data['Brain Weight(grams)'].values
#Y
#X


# In[94]:

#find the mean valu of X and Y
mean_x = np.mean(X)
mean_y = np.mean(Y)
#mean_x
#mean_y


# In[98]:

#So now find the total number of values of X
n = len(X)
#n


# In[38]:

#Calculate the value of b1 and b0,i mean m(coef) and c(intercept) respectivly using the formula...
# y = mx + c or y = c + mx or y = b0 + b1*x or y = b1*x + b0 
# b1 = sum of (X[i]-mean_x)*(Y[i]-mean_y)/sum of (X[i]-mean_x)**2
# b0 = mean_y-(b1 * mean_x)

number = 0
denom = 0
for i in range(n):
    number += (X[i]-mean_x)*(Y[i]-mean_y)
    denom += (X[i]-mean_x)**2
b1 = number/denom
b0 = mean_y-(b1 * mean_x)

#Print cofficients
print(b1,b0)
    


# In[115]:

#Ploting values and regression line

max_x = np.max(X) + 100
min_x = np.min(X) - 100
min_x


# In[106]:

#calculate the line value of X and Y
x = np.linspace(min_x,max_x,1000)
y = b0 + b1*x


# In[107]:

#ploting line
plt.plot(x,y, color='blue', label='Linear Regression')
#ploting scatter points
plt.scatter(X,Y, c='#ef5423', label='Scatter Plot')
plt.xlabel('Head Size')
plt.ylabel('Brain Weight')
plt.legend()
plt.show()


# In[116]:

#now find R2 ,how efficient the linear regession line is ?
tss_t = 0
tss_r = 0
for i in range(n):
    y_pred = b0+b1*X[i]
    tss_r+=(Y[i]-y_pred)**2
    tss_t+=(Y[i]-mean_y)**2
    
r2=1-(tss_r/tss_t)
print(r2)


# In[ ]:

#so we can see the both of the R2 score is same...which is 0.639311719957


# In[110]:

#now we will see the machine learning library which is sklearn.LinearModel
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

X = X.reshape((n,1))
#create model
reg = LinearRegression()
#filtering training data
reg = reg.fit(X,Y)
#Y prediction
Y_pred = reg.predict(X)
#calculating RMSE R2 score
mse = mean_squared_error(Y,Y_pred)
rmse = np.sqrt(mse)
r2_score = reg.score(X,Y)
print(np.sqrt(mse))
print(r2_score)


# In[ ]:

#so we can see the both of the R2 score is same...which is 0.639311719957

