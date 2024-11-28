'''
To analyse: age vs paymethod,

Not usefull: gender vs spent money,


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

    # Group ages into bins
    bins = [10, 20, 30, 40, 50, 65, 100]  # Age bins
    labels = ['10-20', '20-30', '30-40', '40-50', '50-65', '65+']
    age_groups = pd.cut(data_age, bins=bins, labels=labels)

    # Create a DataFrame for easier handling
    df = pd.DataFrame({'AgeGroup': age_groups, 'PayMethod': data_paymethod})

    # Group by AgeGroup and PayMethod, then count occurrences
    grouped_data = df.groupby(['AgeGroup', 'PayMethod']).size().unstack(fill_value=0)

    grouped_data.plot(kind='bar', figsize=(12, 7), width=0.8)
    plt.title('Age vs Payment Method')
    plt.xlabel('Age Group')
    plt.ylabel('Frequency')
    plt.xticks(rotation=45)
    plt.legend(title='Payment Method', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()

def moneyspend_vs_state(dataset, column_id: list[int]):
    """
    Analyze and visualize the relationship between money spent and in which state.

    Keyword arguments:
    |dataset    -- the dataset as a list of lists
    |column_id  -- a list with [age_column_index, payment_method_column_index]
    """
    data_moneySpend = [entry[column_id[0]] for entry in dataset]
    data_state = [entry[column_id[1]] for entry in dataset]

    df = pd.DataFrame({'State': data_state, 'MoneySpent': data_moneySpend})
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


def main():
    headers, dataset = read_csv("data/shopping_trends.csv")
    age_vs_paymethod(dataset, [1, 12])
    moneyspend_vs_state(dataset, [5,6])

if __name__ == '__main__':
    main()
