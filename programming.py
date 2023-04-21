#!/usr/bin/env python
# coding: utf-8

# # Question 1
# Introduction:
# Special thanks to: https://github.com/justmarkham for sharing the dataset and 
# materials.
# Occupations
# Step 1. Import the necessary libraries

# In[1]:


import pandas as pd


# Step 2. Import the dataset from this address.

# In[2]:


url = 'https://raw.githubusercontent.com/justmarkham/DAT8/master/data/u.user'
users = pd.read_csv(url, sep='|')


# Step 3. Assign it to a variable called users

# In[3]:


users


# Step 4. Discover what is the mean age per occupation
# 

# In[4]:


mean_age = users.groupby('occupation').age.mean()
print(mean_age)


# Step 5. Discover the Male ratio per occupation and sort it from the most to the least

# In[5]:


def male_ratio(x):
    if x == 'M':
        return 1
    else:
        return 0

users['male_ratio'] = users['gender'].apply(male_ratio)
male_ratio = users.groupby('occupation').male_ratio.sum() / users.occupation.value_counts() * 100
male_ratio = male_ratio.sort_values(ascending=False)
print(male_ratio)


# Step 6. For each occupation, calculate the minimum and maximum ages

# In[6]:


age_range = users.groupby('occupation').age.agg(['min', 'max'])
print(age_range)


# Step 7. For each combination of occupation and sex, calculate the mean age

# In[7]:


mean_age = users.groupby(['occupation', 'gender']).age.mean()
print(mean_age)


# Step 8. For each occupation present the percentage of women and men

# In[8]:


gender_counts = users.groupby(['occupation', 'gender']).agg({'gender': 'count'})
occupation_counts = users.groupby('occupation').agg('count')
percentage = gender_counts.div(occupation_counts, level='occupation') * 100
percentage.loc[:, 'gender']
print(percentage.loc[:, 'gender'])


# #       QUESTION 2

# Step 1. Import the necessary libraries

# In[7]:


import pandas as pd
import numpy as np


# Step 2. Import the dataset from this address.
# Step 3. Assign it to a variable called euro12.

# In[9]:


euro = pd.read_csv('https://raw.githubusercontent.com/guipsamora/pandas_exercises/master/02_Filtering_%26_Sorting/Euro12/Euro_2012_stats_TEAM.csv')


# In[10]:


euro


# Step 4. Select only the Goal column.

# In[11]:


euro.Goals


# Step 5. How many team participated in the Euro2012?

# In[12]:


euro.Team.nunique()


# Step 6. What is the number of columns in the dataset?

# In[13]:


euro.shape[1]

Step 7. View only the columns Team, Yellow Cards and Red Cards and assign them to a dataframe called discipline
# In[14]:


discipline = euro[['Team', 'Yellow Cards', 'Red Cards']]


# In[15]:


discipline


# Step 8. Sort the teams by Red Cards, then to Yellow Cards

# In[16]:


discipline.sort_values(by=['Red Cards', 'Yellow Cards'], inplace = True)


# In[17]:


discipline


# Step 9. Calculate the mean Yellow Cards given per Team

# In[18]:


discipline['Yellow Cards'].sum() / len(discipline['Yellow Cards'])


# In[19]:


discipline['Yellow Cards'].mean()


# Step 10. Filter teams that scored more than 6 goals

# In[20]:


euro[euro['Goals'] > 6]


# Step 11. Select the teams that start with G

# In[21]:


euro[euro.Team.str.startswith('G')]


# Step 12. Select the first 7 columns

# In[22]:


euro.head(7)


# In[23]:


euro.iloc[: , 0:7]


# Step 13. Select all columns except the last 3.

# In[24]:


euro.iloc[:, :-3]


# Step 14. Present only the Shooting Accuracy from England, Italy and Russia

# In[25]:


#idk


# In[26]:


euro.loc[euro.Team.isin(['England', 'Italy', 'Russia']), ['Team','Shooting Accuracy']]


# # Question 3
# Housing

# Step 1. Import the necessary libraries

# In[15]:


import pandas as pd
import numpy as np


# Step 2. Create 3 different Series, each of length 100, as follows:
# 1.The first a random number from 1 to 4
# 2.The second a random number from 1 to 3
# 3.The third a random number from 10,000 to 30,000

# In[29]:


s1 = pd.Series(np.random.randint(1, high=5, size=100, dtype='l'))
s2 = pd.Series(np.random.randint(1, high=4, size=100, dtype='l'))
s3 = pd.Series(np.random.randint(10000, high=30001, size=100, dtype='l'))

print(s1, s2, s3)


# Step 3. Create a DataFrame by joining the Series by column

# In[30]:


housemkt = pd.concat([s1, s2, s3], axis=1)
housemkt.head()


# Step 4. Change the name of the columns to bedrs, bathrs, price_sqr_meter

# In[31]:


housemkt.rename(columns = {0: 'bedrs', 1: 'bathrs', 2: 'price_sqr_meter'}, inplace=True)
housemkt.head()


# Step 5. Create a one column DataFrame with the values of the 3 Series and assign it to 'bigcolumn'

# In[33]:


# join concat the values
bigcolumn = pd.concat([s1, s2, s3], axis=0)

# it is still a Series, so we need to transform it to a DataFrame
bigcolumn = bigcolumn.to_frame()
print(type(bigcolumn))

bigcolumn


# Step 6. Check if the DataFrame only goes until index 99

# In[20]:


print(len(bigcolumn)) # Output: 300
print(bigcolumn.tail()) # Output: The last row has index 299


# Step 7. Reindex the DataFrame so it goes from 0 to 299

# In[34]:


bigcolumn.reset_index(drop=True, inplace=True)
bigcolumn


# # QUESTION 4

# The data have been modified to contain some missing values, identified by NaN.
# Using pandas should make this exercise easier, in particular for the bonus question.
# 
# You should be able to perform all of these operations without using a for loop or other looping construct.
# 
# 1.The data in 'wind.data' has the following format:

# In[35]:


"""
Yr Mo Dy   RPT   VAL   ROS   KIL   SHA   BIR   DUB   CLA   MUL   CLO   BEL   MAL
61  1  1 15.04 14.96 13.17  9.29   NaN  9.87 13.67 10.25 10.83 12.58 18.50 15.04
61  1  2 14.71   NaN 10.83  6.50 12.62  7.67 11.50 10.04  9.79  9.67 17.54 13.83
61  1  3 18.50 16.88 12.33 10.13 11.17  6.17 11.25   NaN  8.50  7.67 12.75 12.71
"""


# The first three columns are year, month and day. The remaining 12 columns are average windspeeds in knots at 12 locations in Ireland on that day.

# Step 1. Import the necessary libraries

# In[36]:


import pandas as pd
import datetime


# Step 2. Import the dataset from this address
# Step 3. Assign it to a variable called data and replace the first 3 columns by a proper datetime index.

# In[37]:


# parse_dates gets 0, 1, 2 columns and parses them as the index
data_url = 'https://raw.githubusercontent.com/guipsamora/pandas_exercises/master/06_Stats/Wind_Stats/wind.data'
data = pd.read_csv(data_url, sep = "\s+", parse_dates = [[0,1,2]]) 
data.head()


# Step 4. Year 2061? Do we really have data from this year? Create a function to fix it and apply it.

# In[38]:


# The problem is that the dates are 2061 and so on...

# function that uses datetime
def fix_century(x):
  year = x.year - 100 if x.year > 1989 else x.year
  return datetime.date(year, x.month, x.day)

# apply the function fix_century on the column and replace the values to the right ones
data['Yr_Mo_Dy'] = data['Yr_Mo_Dy'].apply(fix_century)

# data.info()
data.head()


# Step 5. Set the right dates as the index. Pay attention at the data type, it should be datetime64[ns].

# In[39]:


# transform Yr_Mo_Dy it to date type datetime64
data["Yr_Mo_Dy"] = pd.to_datetime(data["Yr_Mo_Dy"])

# set 'Yr_Mo_Dy' as the index
data = data.set_index('Yr_Mo_Dy')

data.head()
# data.info()


# Step 6. Compute how many values are missing for each location over the entire record.
# They should be ignored in all calculations below.

# In[40]:


# "Number of non-missing values for each location: "
data.isnull().sum()


# Step 7. Compute how many non-missing values there are in total.

# In[41]:


#number of columns minus the number of missing values for each location
data.shape[0] - data.isnull().sum()

#or

data.notnull().sum()


# Step 8. Calculate the mean windspeeds of the windspeeds over all the locations and all the times.
# A single number for the entire dataset.

# In[42]:


data.sum().sum() / data.notna().sum().sum()


# Step 9. Create a DataFrame called loc_stats and calculate the min, max and mean windspeeds and standard deviations of the windspeeds at each location over all the days
# A different set of numbers for each location.

# In[43]:


data.describe(percentiles=[])


# Step 10. Create a DataFrame called day_stats and calculate the min, max and mean windspeed and standard deviations of the windspeeds across all the locations at each day.
# A different set of numbers for each day.

# In[44]:


# create the dataframe
day_stats = pd.DataFrame()

# this time we determine axis equals to one so it gets each row.
day_stats['min'] = data.min(axis = 1) # min
day_stats['max'] = data.max(axis = 1) # max 
day_stats['mean'] = data.mean(axis = 1) # mean
day_stats['std'] = data.std(axis = 1) # standard deviations

day_stats.head()


# Step 11. Find the average windspeed in January for each location.
# Treat January 1961 and January 1962 both as January.

# In[45]:


data.loc[data.index.month == 1].mean()


# Step 12. Downsample the record to a yearly frequency for each location.

# In[46]:


data.groupby(data.index.to_period('A')).mean()


# Step 13. Downsample the record to a monthly frequency for each location.

# In[47]:


data.groupby(data.index.to_period('M')).mean()


# Step 14. Downsample the record to a weekly frequency for each location.

# In[48]:


data.groupby(data.index.to_period('W')).mean()


# Step 15. Calculate the min, max and mean windspeeds and standard deviations of the windspeeds across all locations for each week (assume that the first week starts on January 2 1961) for the first 52 weeks.

# In[49]:


# resample data to 'W' week and use the functions
weekly = data.resample('W').agg(['min','max','mean','std'])

# slice it for the first 52 weeks and locations
weekly.loc[weekly.index[1:53], "RPT":"MAL"] .head(10)


# # QUESTION 5
# 

# Step 1. Import the necessary libraries

# In[50]:


import pandas as pd


# Step 2. Import the dataset from this address.

# Step 3. Assign it to a variable called chipo.

# In[51]:


chipo = pd.read_csv('https://raw.githubusercontent.com/justmarkham/DAT8/master/data/chipotle.tsv', sep='\t')


# Step 4. See the first 10 entries.

# In[52]:


chipo.head(10)


# Step 5. What is the number of observations in the dataset?

# In[53]:


# Solution 1
chipo.shape[0]


# In[54]:


# Solution 2
chipo.info()


# Step 6. What is the number of columns in the dataset?

# In[55]:


chipo.shape[1]


# Step 7. Print the name of all the columns.

# In[56]:


chipo.columns.values


# Step 8. How is the dataset indexed?

# In[57]:


chipo.index


# The dataset is indexed using a RangeIndex object.

# Step 9. Which was the most-ordered item?

# In[58]:


chipo.groupby(['item_name']).quantity.sum().sort_values(ascending = False).index[0]


# In[59]:


chipo.groupby('item_name').sum().sort_values(['quantity'], ascending=False).head(1)


# The most-ordered item is the Chicken Bowl.

# Step 10. For the most-ordered item, how many items were ordered?

# In[60]:


chipo.groupby(['item_name']).quantity.sum().sort_values(ascending = False).values[0]


# In[61]:


chipo.groupby('item_name').sum().sort_values(['quantity'], ascending=False).head(1)


# 761 items of the Chicken Bowl were ordered.

# Step 11. What was the most ordered item in the choice_description column?

# In[62]:


chipo.groupby(['choice_description']).quantity.sum().sort_values(ascending = False).index[0]


# In[63]:


chipo.groupby('choice_description').sum().sort_values(['quantity'], ascending=False).head(1)


# The most-ordered item in the choice_description column is Diet Coke.

# Step 12. How many items were ordered in total?

# In[64]:


chipo.quantity.sum()


# There were 4972 items ordered in total.

# Step 13.Turn the item price into a float
# 13(a) Check the item price type

# In[65]:


chipo.dtypes.item_price


# Step 13.b. Create a lambda function and change the type of item price

# In[66]:


chipo.item_price = chipo.item_price.apply(lambda x: float(x[1:]))


# Step 13.c. Check the item price type

# In[67]:


chipo.item_price.dtypes


# Step 14. How much was the revenue for the period in the dataset?

# In[68]:


chipo['revenue'] = chipo['quantity']*chipo.item_price
total_revenue = chipo.revenue.sum()
total_revenue


# The revenue for the period in the dataset was $39,237.02.

# Step 15. How many orders were made in the period?

# In[69]:


total_order = chipo.order_id.nunique()
total_order


# In[70]:


total_order = chipo.order_id.nunique()
total_order


# Step 16. What is the average revenue amount per order?

# In[71]:


total_revenue / total_order


# In[72]:


# Solution 1
#chipo['revenue'] = chipo['quantity'] * chipo['item_price']
order_grouped = chipo.groupby(by=['order_id']).sum()
order_grouped.mean()['revenue']


# In[73]:


# Solution 2
chipo.groupby(by=['order_id']).sum().mean()['revenue']


# Step 17. How many different items are sold?

# In[74]:


chipo.item_name.nunique()


# In[75]:


chipo.item_name.value_counts().count()


# # QUESTION 6 
# Create a line plot showing the number of marriages and divorces per capita in the U.S. between 1867 and 2014. Label both of the lines and show the legend.
# 
# Don't forget to label your axes!
# 
# Bonus: Use the ggplot style.

# In[96]:


import pandas as pd

us_marriage_divorce_data = pd.read_csv('data/us-marriages-divorces-1867-2014 (2).csv')
years = us_marriage_divorce_data['Year'].values
marriages_per_capita = us_marriage_divorce_data['Marriages_per_1000'].values
divorces_per_capita = us_marriage_divorce_data['Divorces_per_1000'].values


# # QUESTION 7
# Create a vertical bar chart comparing the number of marriages and divorces per capita in the U.S. between 1900, 1950, and 2000.
# 
# Don't forget to label your axes!

# In[97]:


import pandas as pd
us_marriage_divorce_data = pd.read_csv('data/us-marriages-divorces-1867-2014.csv')
us_marriage_divorce_data = us_marriage_divorce_data[
    us_marriage_divorce_data['Year'].apply(lambda x: x in [1900, 1950, 2000])]

years = us_marriage_divorce_data['Year'].values
marriages_per_capita = us_marriage_divorce_data['Marriages_per_1000'].values
divorces_per_capita = us_marriage_divorce_data['Divorces_per_1000'].values


# In[ ]:




