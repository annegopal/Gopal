
# coding: utf-8

# In[1]:

from scipy import stats


# In[2]:

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')


# In[6]:

import pandas_datareader as web


# In[10]:

# Start and End Dates for the Data
start = pd.to_datetime(2010-1-4)
end = pd.to_datetime('today')


# In[14]:

spy_etf = web.DataReader('SPY','google',start,end)
aapl = web.DataReader('AAPL','google',start,end)


# In[15]:

spy_etf.head()


# In[19]:

aapl['Close'].plot(label='AAPL',figsize = (10,8))
spy_etf['Close'].plot(label='SPY Index')
plt.legend()
plt.show()


# In[20]:

# Comparig the Cumulative Returns
aapl['Cumulative'] = aapl['Close']/aapl['Close'].iloc[0]
spy_etf['Cumulative'] = spy_etf['Close']/spy_etf['Close'].iloc[0]


# In[21]:

aapl['Cumulative'].plot(label='AAPL Cumulative',figsize = (10,8))
spy_etf['Cumulative'].plot(label='SPY Cumulative')
plt.legend()
plt.show()


# In[22]:

# Comparing Daily Returns
aapl['Daily Return'] = aapl['Close'].pct_change(1)
spy_etf['Daily Return'] = spy_etf['Close'].pct_change(1)


# In[24]:

# Scatter plot to visually inspect the correlation
plt.scatter(aapl['Daily Return'],spy_etf['Daily Return'],alpha=0.3)
plt.show()


# In[26]:

# Finding the actual beta(correlation) using Linear Regression
beta,alpha,r_value,p_value,std_err = stats.linregress(aapl['Daily Return'].iloc[1:],spy_etf['Daily Return'].iloc[1:])


# In[27]:

beta


# In[28]:

alpha


# In[29]:

r_value


# In[30]:

"""
Creating some noise to be correlated to SPY
"""
noise = np.random.normal(0,0.001,len(spy_etf['Daily Return'].iloc[1:]))


# In[31]:

noise


# In[32]:

spy_etf['Daily Return'].iloc[1:] + noise


# In[35]:

fake_stock = spy_etf['Daily Return'].iloc[1:] + noise


# In[38]:

plt.scatter(fake_stock,spy_etf['Daily Return'].iloc[1:],alpha=0.3)
plt.show()


# In[39]:

# Finding the actual beta(correlation) using Linear Regression
beta,alpha,r_value,p_value,std_err = stats.linregress(fake_stock,spy_etf['Daily Return'].iloc[1:])


# In[40]:

beta


# In[41]:

alpha


# In[42]:

r_value


# In[ ]:



