
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


import seaborn as sns


# In[3]:


from plotly import __version__


# In[4]:


import cufflinks as cf


# In[5]:


from plotly.offline import download_plotlyjs,init_notebook_mode,plot,iplot


# In[6]:


init_notebook_mode(connected=True)


# In[7]:


cf.go_offline()


# In[8]:


import pandas_datareader
import datetime


# In[9]:


import pandas_datareader.data as web


# In[10]:


import fix_yahoo_finance as yf
yf.pdr_override()


# In[11]:


# Start and End Dates for the Data
start = datetime.datetime(1997,1,1)
end = datetime.date.today()
#end = pd.to_datetime('today')
#end = datetime.datetime(2017,1,1)


# In[12]:


datetime.MAXYEAR


# In[13]:


tickers = ['SHW','AOS','TJX','ROST','ALGN','MA'] #'^GSPC','^DJI'
portfolio = pd.DataFrame()
for t in tickers:
    portfolio[t] = web.get_data_yahoo(t,start,end)['Adj Close']


# In[14]:


portfolio.info()


# In[15]:


portfolio.head()


# In[16]:


# Filling the missing values
portfolio.fillna(method='ffill',inplace=True)


# In[17]:


portfolio.head()


# In[18]:


portfolio.iplot(title = 'Portfolio',xTitle='Year',yTitle='Price',theme='pearl')
plt.show();


# In[19]:


# Calculating Daily Returns
Daily_Returns = portfolio.pct_change(1)
Daily_Returns.plot(figsize = (16,8))
plt.show();


# In[20]:


# Calculating Cumulative Daily Returns
Cumulative_Returns = (1 + Daily_Returns).cumprod()
Cumulative_Returns.iplot(title = 'Cumulative Returns',xTitle='Year',yTitle='Price',theme='pearl')
plt.show();


# # Total Return of Portfolio

# In[21]:


returns = portfolio.pct_change(1)
#returns = (portfolio/portfolio.shift(1))-1
returns.head()


# In[22]:


weights = np.array([.166,.166,.166,.166,.166,.166])


# In[23]:


np.dot(returns, weights)


# In[24]:


annual_returns = returns.mean() * 250
annual_returns


# In[25]:


np.dot(annual_returns, weights)


# In[26]:


portfolio_1 = str(round(np.dot(annual_returns, weights),5)*100)+'%'
print(portfolio_1)

