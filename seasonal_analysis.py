import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mlxtend.preprocessing import TransactionEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_selection import RFE
from itertools import combinations

# read the data from Bakery.csv
data = pd.read_csv('Bakery.csv')

# extract the season from DateTime format i.e.(2016-05-11 13:23:34) to (Spring)
data['Season'] = pd.to_datetime(data['DateTime']).dt.quarter

# group transactions by transaction number and aggregate the items & season
transactions_with_season = data.groupby('TransactionNo').agg({'Items': list, 'Season': 'first'})

# find the season with the minimum number of transactions
min_transactions = transactions_with_season['Season'].value_counts().min()

# make a new dataframe to store season-normalized data
normalized_transactions_with_season = pd.DataFrame(columns=transactions_with_season.columns)

# get equal amount of transactions for each season
for season in [1, 2, 3, 4]:
    season_data = transactions_with_season[transactions_with_season['Season'] == season]
    normalized_season_data = season_data.sample(min_transactions, random_state=250)
    normalized_transactions_with_season = pd.concat([normalized_transactions_with_season, normalized_season_data])

# one-hot encode the non-normalized items
te = TransactionEncoder()
te_ary = te.fit(transactions_with_season['Items']).transform(transactions_with_season['Items'])
encoded_items = pd.DataFrame(te_ary, columns=te.columns_)

# one-hot encode the normalized items
te = TransactionEncoder()
te_ary_balanced = te.fit(normalized_transactions_with_season['Items']).transform(normalized_transactions_with_season['Items'])
encoded_items_normalized = pd.DataFrame(te_ary_balanced, columns=te.columns_)

# split data into training/testing sets
X_train, X_test, y_train, y_test = train_test_split(encoded_items, transactions_with_season['Season'], test_size=0.2, random_state=250)

# split data into training/testing sets for balanced data
X_train_normalized, X_test_normalized, y_train_normalized, y_test_normalized = train_test_split(encoded_items_normalized, normalized_transactions_with_season['Season'], test_size=0.2, random_state=250)

# reset index of X_test
X_test.reset_index(drop=True, inplace=True)

# reset index of X_test_normalized
X_test_normalized.reset_index(drop=True, inplace=True)

# select features with RFE
tree = DecisionTreeClassifier()
rfe = RFE(estimator=tree, n_features_to_select=90)  
rfe.fit(X_train, y_train)
selected_features = X_train.columns[rfe.support_]

# train decision tree with selected features
tree_selected_features = DecisionTreeClassifier()
tree_selected_features.fit(X_train[selected_features], y_train)

# convert y_train_normalized to int32 so it can be used in the DecisionTreeClassifier
y_train_normalized = y_train_normalized.astype('int32')

# convert y_test_normalized to int32 so it can be used to test the accuracy of the model
y_test_normalized = y_test_normalized.astype('int32')

# train decision tree with normalized data
normalized_tree = DecisionTreeClassifier(max_depth=10, min_samples_split=5, random_state=250)
normalized_tree.fit(X_train_normalized, y_train_normalized)

# predict season for each itemset in test data
predicted_seasons = tree_selected_features.predict(X_test[selected_features])

# predict season for each itemset in normalized test data
normalized_predicted_seasons = normalized_tree.predict(X_test_normalized)

# get accuracy of the pre-normalized model
accuracy = tree_selected_features.score(X_test[selected_features], y_test)
print("Accuracy:", accuracy)

# get accuracy of the normalized model
normalized_accuracy = normalized_tree.score(X_test_normalized, y_test_normalized)
print("Normalized Accuracy:", normalized_accuracy)

# combine predictions with test data
combined_data = pd.DataFrame({
    'Items': X_test.apply(lambda row: ','.join(X_test.columns[row == 1]), axis=1),
    'Season': y_test.values,
    'Predicted_Season': predicted_seasons
})

# combine normalized predictions with normalized test data
normalized_combined_data = pd.DataFrame({
    'Items': X_test_normalized.apply(lambda row: ','.join(X_test_normalized.columns[row == 1]), axis=1),
    'Season': y_test_normalized.values,
    'Predicted_Season': normalized_predicted_seasons
})

# split items column into strings and get the number of items in each transaction
combined_data['Num_Items'] = combined_data['Items'].str.split(',').apply(len)

# filter out itemsets with only one item
combined_data_filtered = combined_data[combined_data['Num_Items'] > 1]

# drop 'Num_Items' column after filtering
combined_data_filtered.drop(columns=['Num_Items'], inplace=True)

# make dataframe from predicted seasons
predicted_seasons_df = pd.DataFrame(predicted_seasons, index=combined_data.index, columns=['Predicted_Season'])

# filter predicted seasons to match the filtered test data
predicted_seasons_filtered = predicted_seasons_df.loc[combined_data_filtered.index]

# combine predicted seasons with test data
combined_data_filtered['Predicted_Season'] = predicted_seasons_filtered

#print("combined_data_filtered: ", combined_data_filtered)

# ---------------------------------------------------------------------------------------------------------

# split items column into strings and get the number of items in each transaction for normalized data
normalized_combined_data['Num_Items'] = normalized_combined_data['Items'].str.split(',').apply(len)

# filter out itemsets with only one item
normalized_combined_data_filtered = normalized_combined_data[normalized_combined_data['Num_Items'] > 1]

# drop 'Num_Items' column after filtering
normalized_combined_data_filtered.drop(columns=['Num_Items'], inplace=True)

# make dataframe from predicted seasons
normalized_predicted_seasons_df = pd.DataFrame(normalized_predicted_seasons, index=normalized_combined_data.index, columns=['Predicted_Season'])

# filter predicted seasons to match the filtered test data
normalized_predicted_seasons_filtered = normalized_predicted_seasons_df.loc[normalized_combined_data_filtered.index]

# combine predicted seasons with test data
normalized_combined_data_filtered['Predicted_Season'] = normalized_predicted_seasons_filtered

#print("normalized_combined_data_filtered: ", normalized_combined_data_filtered)

# store counts of itemsets for each predicted season in a dictionary
predicted_season_itemset_counts = {}

# iterate over predicted seasons
for predicted_season in [1,2,3,4]:
    # filter combined data to the current season
    seasonal_data = combined_data_filtered[combined_data_filtered['Predicted_Season'] == predicted_season]
    
    # store counts of itemsets for current predicted season in a dictionary
    itemset_counts = {}
    
    # iterate over each itemset in filtered data
    for itemset in seasonal_data['Items']:
        # split itemset string into list of items
        items_list = itemset.split(',')
    
        # iterate over each combination of items in itemset
        for r in range(2, len(items_list) + 1):  # start combinations at 2 items
            for combination in combinations(items_list, r):
                # convert combination to tuple
                combination_tuple = tuple(combination)
            
                # increase count for combination in itemset counts
                if combination_tuple not in itemset_counts:
                    itemset_counts[combination_tuple] = 0
                itemset_counts[combination_tuple] += 1
    
    # sort itemset counts dictionary by count
    sorted_itemset_counts = {k: v for k, v in sorted(itemset_counts.items(), key=lambda item: item[1], reverse=True)}
    
    # store sorted itemset counts for current predicted season
    predicted_season_itemset_counts[predicted_season] = sorted_itemset_counts
    
top10_predicted_itemsets = []
# print top 10 itemsets for each predicted season
for predicted_season, itemset_counts in predicted_season_itemset_counts.items():
    #print(f"Predicted Season: {predicted_season}")
    counter = 0
    for itemset, count in itemset_counts.items():
        top10_predicted_itemsets.append((itemset,predicted_season, count))
        #print(f"Itemset: {itemset}, Count: {count}")
        counter += 1
        if counter == 10:
            break
    
season_itemset_counts = {}

# iterate over actual seasons
for season in [1,2,3,4]:
    # filter combined data to the current season
    seasonal_data = combined_data_filtered[combined_data_filtered['Season'] == season]
    
    # store counts of itemsets for current season in a dictionary
    itemset_counts = {}
    
    # iterate over each itemset in filtered data
    for itemset in seasonal_data['Items']:
        # split itemset string into list of items
        items_list = itemset.split(',')
    
        # iterate over each combination of items in itemset
        for r in range(2, len(items_list) + 1):  # start combinations at 2 items
            for combination in combinations(items_list, r):
                # convert combination to tuple
                combination_tuple = tuple(combination)
            
                # increase count for combination in itemset counts
                if combination_tuple not in itemset_counts:
                    itemset_counts[combination_tuple] = 0
                itemset_counts[combination_tuple] += 1
    
    # sort itemset counts dictionary by count
    sorted_itemset_counts = {k: v for k, v in sorted(itemset_counts.items(), key=lambda item: item[1], reverse=True)}
    
    # store sorted itemset counts for current season
    season_itemset_counts[season] = sorted_itemset_counts

top10_itemsets = []
# print top 10 itemsets for each actual season
for season, itemset_counts in season_itemset_counts.items():
    #print(f"Actual Season: {season}")
    counter = 0
    for itemset, count in itemset_counts.items():
        top10_itemsets.append((itemset,season, count))
        #print(f"Itemset: {itemset}, Count: {count}")
        counter += 1
        if counter == 10:
            break

# function for plotting top itemsets for a season
def plot_top_itemsets(subplot, predicted_itemsets, itemsets, season):
    predicted_itemset_labels = [str(itemset) for itemset, _, _ in predicted_itemsets]
    itemset_labels = [str(itemset) for itemset, _, _ in itemsets]
    combined_labels = predicted_itemset_labels + itemset_labels
    itemset_values = [count for _, _, count in itemsets]
    predicted_itemset_values = [count for _, _, count in predicted_itemsets]
    
    # get range for x locations
    index = np.arange(20)
    
    # set bar width
    bar_width = 0.35

    # plot bars for predicted values
    subplot.bar(index[:10], predicted_itemset_values, bar_width, label='Predicted')

    # plot bars for actual values
    subplot.bar(index[10:], itemset_values, bar_width, label='Actual')
        
    # set x-axis labels
    subplot.set_xticks(index + bar_width / 2)
    subplot.set_xticklabels(combined_labels, rotation=30, ha='right', fontsize=6)

    if season == 1:
        subplot.set_title("Top 10 Itemsets for Spring")
    elif season == 2:
        subplot.set_title("Top 10 Itemsets for Summer")
    elif season == 3:
        subplot.set_title("Top 10 Itemsets for Fall")
    elif season == 4:
        subplot.set_title("Top 10 Itemsets for Winter")


# create subplots for each season
window, subplots = plt.subplots(2, 2, figsize=(12, 7))

# get itemsets for each season and plot
for plotted_season, subplot in zip([1, 2, 3, 4], subplots.flat):
    actual_seasonal_itemsets = [itemset for itemset in top10_itemsets if itemset[1] == plotted_season]
    predicted_seasonal_itemsets = [itemset for itemset in top10_predicted_itemsets if itemset[1] == plotted_season]
    plot_top_itemsets(subplot, predicted_seasonal_itemsets, actual_seasonal_itemsets, plotted_season)
    subplot.legend()

window.subplots_adjust(hspace=1)
window.tight_layout()
plt.show()

# store counts of itemsets for each predicted season in a dictionary
normalized_predicted_season_itemset_counts = {}

# iterate over predicted seasons
for predicted_season in [1,2,3,4]:
    # filter combined data to the current season
    seasonal_data = normalized_combined_data_filtered[normalized_combined_data_filtered['Predicted_Season'] == predicted_season]
    
    # store counts of itemsets for current predicted season in a dictionary
    itemset_counts = {}
    
    # iterate over each itemset in filtered data
    for itemset in seasonal_data['Items']:
        # split itemset string into list of items
        items_list = itemset.split(',')
    
        # iterate over each combination of items in itemset
        for r in range(2, len(items_list) + 1):  # start combinations at 2 items
            for combination in combinations(items_list, r):
                # convert combination to tuple
                combination_tuple = tuple(combination)
            
                # increase count for combination in itemset counts
                if combination_tuple not in itemset_counts:
                    itemset_counts[combination_tuple] = 0
                itemset_counts[combination_tuple] += 1
    
    # sort itemset counts dictionary by count
    sorted_itemset_counts = {k: v for k, v in sorted(itemset_counts.items(), key=lambda item: item[1], reverse=True)}
    
    # store sorted itemset counts for current predicted season
    normalized_predicted_season_itemset_counts[predicted_season] = sorted_itemset_counts
    
normalized_top10_predicted_itemsets = []
# print top 10 itemsets for each predicted season
for predicted_season, itemset_counts in normalized_predicted_season_itemset_counts.items():
    #print(f"Predicted Season: {predicted_season}")
    counter = 0
    for itemset, count in itemset_counts.items():
        normalized_top10_predicted_itemsets.append((itemset, predicted_season, count))
        #print(f"Itemset: {itemset}, Count: {count}")
        counter += 1
        if counter == 10:
            break
    
normalized_season_itemset_counts = {}

# iterate over actual seasons
for season in [1,2,3,4]:
    # filter combined data to the current season
    seasonal_data = normalized_combined_data_filtered[normalized_combined_data_filtered['Season'] == season]
    
    # store counts of itemsets for current season in a dictionary
    itemset_counts = {}
    
    # iterate over each itemset in filtered data
    for itemset in seasonal_data['Items']:
        # split itemset string into list of items
        items_list = itemset.split(',')
    
        # iterate over each combination of items in itemset
        for r in range(2, len(items_list) + 1):  # start combinations at 2 items
            for combination in combinations(items_list, r):
                # convert combination to tuple
                combination_tuple = tuple(combination)
            
                # increase count for combination in itemset counts
                if combination_tuple not in itemset_counts:
                    itemset_counts[combination_tuple] = 0
                itemset_counts[combination_tuple] += 1
    
    # sort itemset counts dictionary by count
    sorted_itemset_counts = {k: v for k, v in sorted(itemset_counts.items(), key=lambda item: item[1], reverse=True)}
    
    # store sorted itemset counts for current season
    normalized_season_itemset_counts[season] = sorted_itemset_counts

normalized_top10_itemsets = []
# print top 10 itemsets for each actual season
for season, itemset_counts in normalized_season_itemset_counts.items():
    #print(f"Actual Season: {season}")
    counter = 0
    for itemset, count in itemset_counts.items():
        normalized_top10_itemsets.append((itemset,season, count))
        #print(f"Itemset: {itemset}, Count: {count}")
        counter += 1
        if counter == 10:
            break
        
# function for plotting top itemsets for a season
def plot_top_itemsets(subplot, predicted_itemsets, itemsets, season):
    predicted_itemset_labels = [str(itemset) for itemset, _, _ in predicted_itemsets]
    itemset_labels = [str(itemset) for itemset, _, _ in itemsets]
    combined_labels = predicted_itemset_labels + itemset_labels
    itemset_values = [count for _, _, count in itemsets]
    predicted_itemset_values = [count for _, _, count in predicted_itemsets]
    
    # get range for x locations
    index = np.arange(20)
    
    # set bar width
    bar_width = 0.35

    # plot bars for predicted values
    subplot.bar(index[:10], predicted_itemset_values, bar_width, label='Predicted')

    # plot bars for actual values
    subplot.bar(index[10:], itemset_values, bar_width, label='Actual')
        
    # set x-axis labels
    subplot.set_xticks(index + bar_width / 2)
    subplot.set_xticklabels(combined_labels, rotation=30, ha='right', fontsize=6)

    if season == 1:
        subplot.set_title("Top 10 Itemsets for Spring")
    elif season == 2:
        subplot.set_title("Top 10 Itemsets for Summer")
    elif season == 3:
        subplot.set_title("Top 10 Itemsets for Fall")
    elif season == 4:
        subplot.set_title("Top 10 Itemsets for Winter")


# create subplots for each season
window, subplots = plt.subplots(2, 2, figsize=(12, 7))

# get itemsets for each season and plot
for plotted_season, subplot in zip([1, 2, 3, 4], subplots.flat):
    actual_seasonal_itemsets = [itemset for itemset in normalized_top10_itemsets if itemset[1] == plotted_season]
    predicted_seasonal_itemsets = [itemset for itemset in normalized_top10_predicted_itemsets if itemset[1] == plotted_season]
    plot_top_itemsets(subplot, predicted_seasonal_itemsets, actual_seasonal_itemsets, plotted_season)
    subplot.legend()

window.subplots_adjust(hspace=1)
window.tight_layout()
plt.show()