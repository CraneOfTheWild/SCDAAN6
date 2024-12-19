'''
To analyse: age vs paymethod,  spent money vs state, items vs seasons,
items vs money spent, age vs money spent,

Not usefull: gender vs spent money, spent money vs season, spent money vs
paymethod, spent money vs frequency of purchases

'''
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from collections import Counter

def read_csv(filename: str) -> tuple[list, list[list]]:
    """
    Read in a csv file into an array.

    Keyword arguments:
    |filename   -- the name of the csv file including the .csv
    """
    pd_dataset = pd.read_csv(filename)
    headers = pd_dataset.columns.to_list()
    dataset = pd_dataset.to_numpy()
    return headers, dataset

def age_vs_paymethod(dataset, column_id: list[int]):
    """
    Analyze and visualize the relationship between age and payment method.

    Keyword arguments:
    |dataset    -- the dataset as a list of lists
    |column_id  -- a list with [age_column_index, payment_method_column_index]
    """
    data_age = [entry[column_id[0]] for entry in dataset]
    data_paymethod = [entry[column_id[1]] for entry in dataset]

    bins = [10, 20, 30, 40, 50, 65, 100]
    labels = ['10-20', '20-30', '30-40', '40-50', '50-65', '65+']
    age_groups = pd.cut(data_age, bins=bins, labels=labels)

    df = pd.DataFrame({'AgeGroup': age_groups, 'PayMethod': data_paymethod})
    grouped_data = df.groupby(['AgeGroup', 'PayMethod'], observed=False).size().unstack(fill_value=0)

    grouped_data.plot(kind='bar', figsize=(12, 7), width=0.8)
    plt.title('Age vs Payment Method')
    plt.xlabel('Age Group')
    plt.ylabel('Number of Persons')
    plt.xticks(rotation=45)
    plt.legend(title='Payment Method', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()

def moneyspent_vs_state(dataset, column_id: list[int]):
    """
    Analyze and visualize the relationship between money spent and in which state.

    Keyword arguments:
    |dataset    -- the dataset as a list of lists
    |column_id  -- a list with [age_column_index, payment_method_column_index]
    """
    data_moneySpent = [entry[column_id[0]] for entry in dataset]
    data_state = [entry[column_id[1]] for entry in dataset]

    df = pd.DataFrame({'State': data_state, 'MoneySpent': data_moneySpent})
    df['MoneySpent'] = pd.to_numeric(df['MoneySpent'], errors='coerce')
    df = df.dropna()

    total_spend = df.groupby('State')['MoneySpent'].sum()
    avg_spend = df.groupby('State')['MoneySpent'].mean()

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))  # 1 row, 2 columns
    total_spend.sort_values(ascending=False).plot(kind='bar', ax=axes[0], color='skyblue')
    axes[0].set_title('Total Money Spent by State')
    axes[0].set_xlabel('State')
    axes[0].set_ylabel('Total Money Spent (USD)')
    axes[0].tick_params(axis='x', rotation=90)

    avg_spend.sort_values(ascending=False).plot(kind='bar', ax=axes[1], color='orange')
    axes[1].set_title('Average Money Spent by State')
    axes[1].set_xlabel('State')
    axes[1].set_ylabel('Average Money Spent (USD)')
    axes[1].tick_params(axis='x', rotation=90)

    plt.tight_layout()
    plt.show()

def item_vs_season(dataset, column_id: list[int]):
    """
    Analyze and visualize the relationship between item bought and in which season.

    Keyword arguments:
    |dataset    -- the dataset as a list of lists
    |column_id  -- a list with [age_column_index, payment_method_column_index]
    """
    data_item = [entry[column_id[0]] for entry in dataset]
    data_season = [entry[column_id[1]] for entry in dataset]

    seasons = ['Spring', 'Summer', 'Fall', 'Winter']
    df = pd.DataFrame({'Item': data_item, 'Season': data_season})
    df['Season'] = pd.Categorical(df['Season'], categories=seasons, ordered=True)
    grouped_data = df.groupby(['Season', 'Item'],observed=False).size().unstack(fill_value=0)

    items = grouped_data.columns.to_list()
    chunks = [items[:6], items[6:12], items[12:19], items[19:]]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for i, chunk in enumerate(chunks):
        grouped_data[chunk].plot(ax=axes[i], kind='line', marker='o')
        axes[i].set_title(f'Items {i * 6 + 1} to {i * 6 + len(chunk)}')
        axes[i].set_xlabel('Season')
        axes[i].set_ylabel('Frequency')
        axes[i].tick_params(axis='x', rotation=45)
        axes[i].legend(title='Item', bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.tight_layout()
    plt.show()

def item_vs_moneyspent(dataset, column_id: list[int]):
    """
    Analyze and visualize the relationship between items bought and total
    and average money spent.

    Keyword arguments:
    |dataset    -- the dataset as a list of lists
    |column_id  -- a list with [item_column_index, month_column_index]
    """

    data_item = [entry[column_id[0]] for entry in dataset]
    data_moneySpent = [entry[column_id[1]] for entry in dataset]

    df = pd.DataFrame({'Item': data_item, 'MoneySpent': data_moneySpent})
    df['MoneySpent'] = pd.to_numeric(df['MoneySpent'], errors='coerce')
    df = df.dropna()

    total_spend = df.groupby('Item')['MoneySpent'].sum()
    avg_spend = df.groupby('Item')['MoneySpent'].mean()

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    total_spend.sort_values(ascending=False).plot(kind='bar', ax=axes[0], color='skyblue')
    axes[0].set_title('Total Money Spent per Item')
    axes[0].set_xlabel('Item')
    axes[0].set_ylabel('Total Money Spent (USD)')
    axes[0].tick_params(axis='x', rotation=45)

    avg_spend.sort_values(ascending=False).plot(kind='bar', ax=axes[1], color='orange')
    axes[1].set_title('Average Money Spent per Item')
    axes[1].set_xlabel('Item')
    axes[1].set_ylabel('Average Money Spent (USD)')
    axes[1].tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.show()

def age_vs_moneyspent(dataset, column_id: list[int]):
    """
    Analyze and visualize the relationship between age and money spent.

    Keyword arguments:
    |dataset    -- the dataset as a list of lists
    |column_id  -- a list with [age_column_index, money_spent_column_index]
    """
    data_age = [entry[column_id[0]] for entry in dataset]
    data_moneySpent = [entry[column_id[1]] for entry in dataset]

    bins = [10, 20, 30, 40, 50, 65, 100]
    labels = ['10-20', '20-30', '30-40', '40-50', '50-65', '65+']
    age_groups = pd.cut(data_age, bins=bins, labels=labels, ordered=True)

    df = pd.DataFrame({'AgeGroup': age_groups, 'MoneySpent': data_moneySpent})
    df['MoneySpent'] = pd.to_numeric(df['MoneySpent'], errors='coerce')
    df = df.dropna()

    total_spend = df.groupby('AgeGroup', observed=False)['MoneySpent'].sum()
    avg_spend = df.groupby('AgeGroup', observed=False)['MoneySpent'].mean()

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    total_spend.plot(kind='bar', ax=axes[0], color='skyblue')
    axes[0].set_title('Total Money Spent per Age Group')
    axes[0].set_xlabel('Age Group')
    axes[0].set_ylabel('Total Money Spent (USD)')
    axes[0].tick_params(axis='x', rotation=45)

    avg_spend.plot(kind='bar', ax=axes[1], color='orange')
    axes[1].set_title('Average Money Spent per Age Group')
    axes[1].set_xlabel('Age Group')
    axes[1].set_ylabel('Average Money Spent (USD)')
    axes[1].tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.show()

def season_vs_moneyspent(dataset, column_id: list[int]):
    """
    Analyze and visualize the relationship between total and
    average money spent and in which season.

    Keyword arguments:
    |dataset    -- the dataset as a list of lists
    |column_id  -- a list with [age_column_index, payment_method_column_index]
    """
    data_moneySpent = [entry[column_id[0]] for entry in dataset]
    data_season = [entry[column_id[1]] for entry in dataset]

    seasons = ['Spring', 'Summer', 'Fall', 'Winter']
    df = pd.DataFrame({'MoneySpent': data_moneySpent, 'Season': data_season})
    df['Season'] = pd.Categorical(df['Season'], categories=seasons, ordered=True)
    df['MoneySpent'] = pd.to_numeric(df['MoneySpent'], errors='coerce')
    df = df.dropna()

    total_spend = df.groupby('Season', observed=False)['MoneySpent'].sum()
    avg_spend = df.groupby('Season', observed=False)['MoneySpent'].mean()

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    total_spend.plot(kind='bar', ax=axes[0], color='skyblue')
    axes[0].set_title('Total Money Spent per Season')
    axes[0].set_xlabel('Season')
    axes[0].set_ylabel('Total Money Spent (USD)')
    axes[0].tick_params(axis='x', rotation=45)

    avg_spend.plot(kind='bar', ax=axes[1], color='orange')
    axes[1].set_title('Average Money Spent per Season')
    axes[1].set_xlabel('Season')
    axes[1].set_ylabel('Average Money Spent (USD)')
    axes[1].tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.show()

def paymethod_vs_moneyspent(dataset,column_id: list[int]):
    """
    Analyze and visualize the relationship between total
    and average money spent per paymethod.

    Keyword arguments:
    |dataset    -- the dataset as a list of lists
    |column_id  -- a list with [age_column_index, payment_method_column_index]
    """
    data_moneySpent = [entry[column_id[0]] for entry in dataset]
    data_paymethod = [entry[column_id[1]] for entry in dataset]

    df = pd.DataFrame({'PayMethod': data_paymethod, 'MoneySpent': data_moneySpent})
    grouped_data = df.groupby(['PayMethod', 'MoneySpent'], observed=False).size().unstack(fill_value=0)

    df['MoneySpent'] = pd.to_numeric(df['MoneySpent'], errors='coerce')
    df = df.dropna()

    total_spend = df.groupby('PayMethod', observed=False)['MoneySpent'].sum()
    avg_spend = df.groupby('PayMethod', observed=False)['MoneySpent'].mean()

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    total_spend.plot(kind='bar', ax=axes[0], color='skyblue')
    axes[0].set_title('Total Money Spent per Pay Method')
    axes[0].set_xlabel('Pay Method')
    axes[0].set_ylabel('Total Money Spent (USD)')
    axes[0].tick_params(axis='x', rotation=45)

    avg_spend.plot(kind='bar', ax=axes[1], color='orange')
    axes[1].set_title('Average Money Spent per Pay Method')
    axes[1].set_xlabel('Pay Method')
    axes[1].set_ylabel('Average Money Spent (USD)')
    axes[1].tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.show()

def moneyspent_vs_spendingrate(dataset,column_id: list[int]):
    """
    Analyze and visualize the relationship between total and
    average money spent per frequency of purchase.

    Keyword arguments:
    |dataset    -- the dataset as a list of lists
    |column_id  -- a list with [age_column_index, payment_method_column_index]
    """
    data_moneySpent = [entry[column_id[0]] for entry in dataset]
    data_spendingRate = [entry[column_id[1]] for entry in dataset]

    df = pd.DataFrame({'SpendingRate': data_spendingRate, 'MoneySpent': data_moneySpent})
    df['MoneySpent'] = pd.to_numeric(df['MoneySpent'], errors='coerce')
    df = df.dropna()

    total_spend = df.groupby('SpendingRate')['MoneySpent'].sum()
    avg_spend = df.groupby('SpendingRate')['MoneySpent'].mean()

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    total_spend.sort_values(ascending=False).plot(kind='bar', ax=axes[0], color='skyblue')
    axes[0].set_title('Total Money Spent vs Frequency of Purchase')
    axes[0].set_xlabel('Frequency of Purchase')
    axes[0].set_ylabel('Total Money Spent (USD)')
    axes[0].tick_params(axis='x', rotation=90)

    avg_spend.sort_values(ascending=False).plot(kind='bar', ax=axes[1], color='orange')
    axes[1].set_title('Average Money Spent vs Frequency of Purchase')
    axes[1].set_xlabel('Frequency of Purchase')
    axes[1].set_ylabel('Average Money Spent (USD)')
    axes[1].tick_params(axis='x', rotation=90)

    plt.tight_layout()
    plt.show()

def main():
    headers, dataset = read_csv("data/shopping_trends.csv")
    '''
    The Functions below are the functions that could be usefull in our
    project.
    '''
    age_vs_paymethod(dataset, [1,12])
    moneyspent_vs_state(dataset, [5,6])
    item_vs_season(dataset, [3,9])
    item_vs_moneyspent(dataset, [3,5])
    age_vs_moneyspent(dataset, [1,5])

    '''
    These functions below are not very usefull for our analysis.
    '''
    moneyspent_vs_spendingrate(dataset, [5,18])
    season_vs_moneyspent(dataset, [5,9])
    paymethod_vs_moneyspent(dataset, [5,12])


if __name__ == '__main__':
    main()
