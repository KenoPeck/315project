# import the data from Bakery.csv to a pandas dataframe

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Read the data from Bakery.csv
data = pd.read_csv('Bakery.csv')

# extract the day from DateTime format i.e.(2016-05-11 13:23:34) to (2016-05-11)
data['Date'] = pd.to_datetime(data['DateTime']).dt.date

# extract the time from DateTime format i.e.(2016-05-11 13:23:34) to (13:23:34)
data['Time'] = pd.to_datetime(data['DateTime']).dt.time

# extract the hour from DateTime format i.e.(2016-05-11 13:23:34) to (13)
data['Hour'] = pd.to_datetime(data['DateTime']).dt.hour

# extract the hour and minute from DateTime format i.e.(2016-05-11 13:23:34) to (13:23)
data['Hour_Minute'] = pd.to_datetime(data['DateTime']).dt.strftime('%H:%M')

# plot how many transactions occured on each date
data['Date'].value_counts().sort_index().plot(kind='bar', figsize=(20,10))
plt.xlabel('Date')
plt.ylabel('Number of Transactions')
plt.title('Number of Transactions on each Date')
plt.show()

# plot how many transactions occured on each hour
data['Hour'].value_counts().sort_index().plot(kind='bar', figsize=(20,10))
plt.xlabel('Hour')
plt.ylabel('Number of Transactions')
plt.title('Number of Transactions on each Hour')
plt.show()

# plot how many transactions occured on each hour and minute
data['Hour_Minute'].value_counts().sort_index().plot(kind='bar', figsize=(20,10))
plt.xlabel('Hour:Minute')
plt.ylabel('Number of Transactions')
plt.title('Number of Transactions on each Hour:Minute')
plt.show()


# plot how many of each item was sold
data['Items'].value_counts().plot(kind='bar', figsize=(20,10))
plt.xlabel('Items')
plt.ylabel('Number of Transactions')
plt.title('Number of Transactions for each Item')
plt.show()

# group by 'Items' and 'DayType' and get the count of transactions
item_daytype_counts = data.groupby(['Items', 'DayType']).size().unstack()

# add a new column for the total number of transactions for each item
item_daytype_counts['Total'] = item_daytype_counts.sum(axis=1)

# sort by the total number of transactions
item_daytype_counts = item_daytype_counts.sort_values('Total', ascending=False)

# drop the 'Total' column as we don't need it anymore
item_daytype_counts = item_daytype_counts.drop(columns='Total')

# plot the number of transactions for each item on weekend vs weekday
item_daytype_counts.plot(kind='bar', stacked=True, figsize=(20,10))
plt.xlabel('Items')
plt.ylabel('Number of Transactions')
plt.title('Number of Transactions for each Item on Weekend vs Weekday')
plt.show()

# do the same with daypart instead of daytype
item_daypart_counts = data.groupby(['Items', 'Daypart']).size().unstack()
item_daypart_counts['Total'] = item_daypart_counts.sum(axis=1)
item_daypart_counts = item_daypart_counts.sort_values('Total', ascending=False)
item_daypart_counts = item_daypart_counts.drop(columns='Total')
item_daypart_counts.plot(kind='bar', stacked=True, figsize=(20,10))
plt.xlabel('Items')
plt.ylabel('Number of Transactions')
plt.title('Number of Transactions for each Item on Different Dayparts')
plt.show()

# create a pie chart of how many items were sold on weekend vs weekday
item_daytype_counts.sum().plot(kind='pie', autopct='%1.1f%%', figsize=(10,10))
plt.title('Percentage of Transactions on Weekend vs Weekday')
plt.show()

# create a pie chart of how many items were sold on different dayparts
item_daypart_counts.sum().plot(kind='pie', autopct='%1.1f%%', figsize=(10,10))
plt.title('Percentage of Transactions on Different Dayparts')
plt.show()

# create a pie chart of how many items were sold each month
# also label each month with the name rather than 0-11

data['Month'] = pd.to_datetime(data['Date']).dt.month
data['Month'] = data['Month'].map({1: 'January', 2: 'February', 3: 'March', 4: 'April', 5: 'May', 6: 'June',
                                   7: 'July', 8: 'August', 9: 'September', 10: 'October', 11: 'November', 12: 'December'})
data['Month'].value_counts().sort_index().plot(kind='pie', autopct='%1.1f%%', figsize=(10,10))
plt.title('Percentage of Transactions on each Month')
plt.show()

# create a pie chart of how many items were sold each day of the week
data['Day'] = pd.to_datetime(data['Date']).dt.dayofweek
data['Day'] = data['Day'].map({0: 'Monday', 1: 'Tuesday', 2: 'Wednesday', 3: 'Thursday', 4: 'Friday', 5: 'Saturday', 6: 'Sunday'})
data['Day'].value_counts().sort_index().plot(kind='pie', autopct='%1.1f%%', figsize=(10,10))
plt.title('Percentage of Transactions on each Day of the Week')
plt.show()

# create a top 10 list of items sold on each day of the week
top10_items = data['Items'].value_counts().head(10).index
top10_items_data = data[data['Items'].isin(top10_items)]
top10_items_data = top10_items_data.groupby(['Day', 'Items']).size().unstack()
top10_items_data = top10_items_data.fillna(0)
top10_items_data = top10_items_data.astype(int)
top10_items_data = top10_items_data.reindex(columns=top10_items)
top10_items_data.plot(kind='bar', stacked=True, figsize=(20,10))
plt.xlabel('Day of the Week')
plt.ylabel('Number of Transactions')
plt.title('Top 10 Items Sold on each Day of the Week')
plt.show()

# create a top 10 list of items sold on each month
top10_items_data = data[data['Items'].isin(top10_items)]
top10_items_data = top10_items_data.groupby(['Month', 'Items']).size().unstack()
top10_items_data = top10_items_data.fillna(0)
top10_items_data = top10_items_data.astype(int)
top10_items_data = top10_items_data.reindex(columns=top10_items)
top10_items_data.plot(kind='bar', stacked=True, figsize=(20,10))
plt.xlabel('Month')
plt.ylabel('Number of Transactions')
plt.title('Top 10 Items Sold on each Month')
plt.show()

# create a histogram of the number of items sold on each transaction
# count how many transactions have each transaction id, and create a histogram
data['TransactionNo'].value_counts().plot(kind='hist', bins=50, figsize=(20,10))
plt.xlabel('Number of Items')
plt.ylabel('Number of Transactions')
plt.title('Number of Items Sold on each Transaction')
plt.show()

# find the 10 most commonly bought item in the same transaction as each of the top 10 items
top10_items_pairs = {}
for item in top10_items:
    item_transactions = data[data['Items'] == item]['TransactionNo']
    item_transactions_data = data[data['TransactionNo'].isin(item_transactions)]
    item_transactions_data = item_transactions_data[item_transactions_data['Items'] != item]
    top10_items_pairs[item] = item_transactions_data['Items'].value_counts().head(10)
# plot the results
fig, ax = plt.subplots(5, 2, figsize=(20,10))
for i, item in enumerate(top10_items_pairs):
    top10_items_pairs[item].plot(kind='bar', ax=ax[i//2, i%2])
    ax[i//2, i%2].set_title(item)
plt.tight_layout()
plt.show()

# find the confidence of each pair of items in the whole dataset
pairs = {}
for item in data['Items'].unique():
    item_transactions = data[data['Items'] == item]['TransactionNo']
    item_transactions_data = data[data['TransactionNo'].isin(item_transactions)]
    item_transactions_data = item_transactions_data[item_transactions_data['Items'] != item]
    pairs[item] = item_transactions_data['Items'].value_counts()
# calculate the confidence
confidence = {}
for item in pairs:
    confidence[item] = pairs[item] / data['Items'].value_counts()[item]
# plot the top 10 pairs of items with the highest confidence in a bar chart
top10_confidence = {}
for item in confidence:
    top10_confidence[item] = confidence[item].sort_values(ascending=False).head(10)
# Limit the number of items from top10_confidence to 10
limited_top10_confidence = dict(list(top10_confidence.items())[:10])



for i, item in enumerate(limited_top10_confidence):
    fig, ax = plt.subplots(figsize=(10,5))
    limited_top10_confidence[item].plot(kind='bar', ax=ax)
    ax.set_title(item)
    plt.tight_layout()
    if i < 10:
        plt.show()

# find the lift of each pair of items in the whole dataset
lift = {}
for item in pairs:
    lift[item] = pairs[item] / (data['Items'].value_counts()[item] * data['Items'].value_counts())
# plot the top 10 pairs of items with the highest lift in a bar chart
top10_lift = {}
for item in lift:
    top10_lift[item] = lift[item].sort_values(ascending=False).head(10)
# Limit the number of items from top10_lift to 10
limited_top10_lift = dict(list(top10_lift.items())[:10])


for i, item in enumerate(limited_top10_lift):
    fig, ax = plt.subplots(figsize=(10,5))
    limited_top10_lift[item].plot(kind='bar', ax=ax)
    ax.set_title(item)
    plt.tight_layout()
    if i < 10:
        plt.show()

# find the support of each pair of items in the whole dataset
support = {}
for item in pairs:
    support[item] = pairs[item] / len(data)
# plot the top 10 pairs of items with the highest support in a bar chart
top10_support = {}
for item in support:
    top10_support[item] = support[item].sort_values(ascending=False).head(10)
# Limit the number of items from top10_support to 10
# each getting their own plot
limited_top10_support = dict(list(top10_support.items())[:10])

for i, item in enumerate(limited_top10_support):
    fig, ax = plt.subplots(figsize=(10,5))
    limited_top10_support[item].plot(kind='bar', ax=ax)
    ax.set_title(item)
    plt.tight_layout()
    if i < 10:
        plt.show()











