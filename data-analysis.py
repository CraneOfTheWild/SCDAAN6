
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from collections import Counter
import seaborn as sns
from scipy.optimize import curve_fit


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
    duplicated_grouped_data = grouped_data.copy()
    duplicated_grouped_data.index = duplicated_grouped_data.index.map(lambda x: f"{x}_2")

    extended_grouped_data = pd.concat([grouped_data, duplicated_grouped_data])

    items = extended_grouped_data.columns.to_list()
    chunks = [items[:6], items[6:12], items[12:19], items[19:]]


    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for i, chunk in enumerate(chunks):

        extended_grouped_data[chunk].plot(ax=axes[i], kind='line', marker='o')
        axes[i].set_title(f'Items {i * 6 + 1} to {i * 6 + len(chunk)}')
        axes[i].set_xlabel('Season')
        axes[i].set_ylabel('Number of Purchases')
        axes[i].tick_params(axis='x', rotation=45)
        axes[i].legend(title='Item', bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(14, 8))
    sns.heatmap(grouped_data, annot=True, fmt="d", cmap='coolwarm', linewidths=0.5)
    plt.title('Heatmap: Frequency of Items Bought by Season')
    plt.xlabel('Item')
    plt.ylabel('Season')
    plt.xticks(rotation=45)
    plt.show()

def sinusoid(x,a,b,phi,c):
    y = a*np.sin(b*x+phi)+c
    return y

def linear_regression(x,y):
    """
    Perform simple linear regression.

    Arguments:
    x -- List of independent variable values.
    y -- List of dependent variable values.

    Returns:
    slope (m), intercept (b), and predicted values based on the regression line.
    """
    n = len(x)
    mean_x = sum(x) / n
    mean_y = sum(y) / n

    numerator = sum((x[i] - mean_x) * (y[i] - mean_y) for i in range(n))
    denominator = sum((x[i] - mean_x) ** 2 for i in range(n))

    slope = numerator / denominator
    intercept = mean_y - slope * mean_x

    predictions = [slope * xi + intercept for xi in x]
    return slope, intercept, predictions

def jacket_boots_regression(dataset, column_id: list[int]):
    '''
    This function performs a regression on the items jacket and boots. The
    boots item needs a linear regression while the jacket items needs a
    sinusoid regression. We first start with a regression for one year.
    '''
    data_item = [entry[column_id[0]] for entry in dataset]
    data_season = [entry[column_id[1]] for entry in dataset]

    seasons = ['Spring', 'Summer', 'Fall', 'Winter']
    df = pd.DataFrame({'Item': data_item, 'Season': data_season})
    df['Season'] = pd.Categorical(df['Season'], categories=seasons, ordered=True)
    grouped_data = df.groupby(['Season', 'Item'],observed=False).size().unstack(fill_value=0)

    season_labels = grouped_data.index.str.replace('_2', '')
    numeric_seasons = list(range(1, len(season_labels) + 1))

    # for jacket
    jacket_data = grouped_data['Jacket']
    initial_guess = [(max(jacket_data) - min(jacket_data)) / 2,  2 * np.pi / 4,
    0,  np.mean(jacket_data)]
    numeric_seasons = np.array(numeric_seasons) / max(numeric_seasons)
    popt, _ = curve_fit(sinusoid, numeric_seasons, jacket_data, p0=initial_guess,
    bounds=([0, 0, -np.pi, min(jacket_data)], [np.inf, np.inf, np.pi, max(jacket_data)]))
    jacket_pred = sinusoid(np.array(numeric_seasons), *popt)

    # for boots
    boots_data = grouped_data['Boots']
    boots_slope, boots_intercept, boots_pred = linear_regression(numeric_seasons, boots_data.tolist())

    plt.figure(figsize=(10, 6))
    plt.plot(numeric_seasons, jacket_data, label='Jacket (Actual)', marker='o', color='blue')
    plt.plot(numeric_seasons, jacket_pred, label='Jacket (Predicted)', linestyle='--', color='blue')
    plt.plot(numeric_seasons, boots_data, label='Boots (Actual)', marker='o', color='orange')
    plt.plot(numeric_seasons, boots_pred, label='Boots (Predicted)', linestyle='--', color='orange')

    plt.title('Regression Analysis: Jacket and Boots')
    plt.xlabel('Season (Numeric)')
    plt.ylabel('Number of Purchases')
    plt.xticks(ticks=numeric_seasons, labels=season_labels, rotation=45)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def hat_jeans_regression(dataset, column_id: list[int]):
    '''
    This function performs a regression on the items hat and jeans. The
    jeans item needs a linear regression while the hat items needs a
    sinusoid regression. We first start with a regression for one year.
    '''
    data_item = [entry[column_id[0]] for entry in dataset]
    data_season = [entry[column_id[1]] for entry in dataset]

    seasons = ['Spring', 'Summer', 'Fall', 'Winter']
    df = pd.DataFrame({'Item': data_item, 'Season': data_season})
    df['Season'] = pd.Categorical(df['Season'], categories=seasons, ordered=True)
    grouped_data = df.groupby(['Season', 'Item'],observed=False).size().unstack(fill_value=0)

    season_labels = grouped_data.index.str.replace('_2', '')
    numeric_seasons = list(range(1, len(season_labels) + 1))

    # for hat
    hat_data = grouped_data['Hat']
    initial_guess = [(max(hat_data) - min(hat_data)) / 2,  2 * np.pi / 4,
    0,  np.mean(hat_data)]
    numeric_seasons = np.array(numeric_seasons) / max(numeric_seasons)
    popt, _ = curve_fit(sinusoid, numeric_seasons, hat_data, p0=initial_guess,
    bounds=([0, 0, -np.pi, min(hat_data)], [np.inf, np.inf, np.pi, max(hat_data)]))
    hat_pred = sinusoid(np.array(numeric_seasons), *popt)

    # for jeans
    jeans_data = grouped_data['Jeans']
    jeans_slope, jeans_intercept, jeans_pred = linear_regression(numeric_seasons, jeans_data.tolist())

    plt.figure(figsize=(10, 6))
    plt.plot(numeric_seasons, hat_data, label='Hat (Actual)', marker='o', color='blue')
    plt.plot(numeric_seasons, hat_pred, label='Hat (Predicted)', linestyle='--', color='blue')
    plt.plot(numeric_seasons, jeans_data, label='Jeans (Actual)', marker='o', color='orange')
    plt.plot(numeric_seasons, jeans_pred, label='Jeans (Predicted)', linestyle='--', color='orange')

    plt.title('Regression Analysis: Hat and Jeans')
    plt.xlabel('Season (Numeric)')
    plt.ylabel('Number of Purchases')
    plt.xticks(ticks=numeric_seasons, labels=season_labels, rotation=45)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def pants_shoes_regression(dataset, column_id: list[int]):
    '''
    This function performs a regression on the items pants and shoes. Both items
    need sinusoid regression. We first start with a regression for one year.
    '''
    data_item = [entry[column_id[0]] for entry in dataset]
    data_season = [entry[column_id[1]] for entry in dataset]

    seasons = ['Spring', 'Summer', 'Fall', 'Winter']
    df = pd.DataFrame({'Item': data_item, 'Season': data_season})
    df['Season'] = pd.Categorical(df['Season'], categories=seasons, ordered=True)
    grouped_data = df.groupby(['Season', 'Item'],observed=False).size().unstack(fill_value=0)

    season_labels = grouped_data.index.str.replace('_2', '')
    numeric_seasons = list(range(1, len(season_labels) + 1))

    # for pants
    pants_data = grouped_data['Pants']
    initial_guess = [(max(pants_data) - min(pants_data)) / 2,  2 * np.pi / 4,
    0,  np.mean(pants_data)]
    numeric_seasons = np.array(numeric_seasons) / max(numeric_seasons)
    popt, _ = curve_fit(sinusoid, numeric_seasons, pants_data, p0=initial_guess,
    bounds=([0, 0, -np.pi, min(pants_data)], [np.inf, np.inf, np.pi, max(pants_data)]))
    pants_pred = sinusoid(np.array(numeric_seasons), *popt)

    # for shoes
    shoes_data = grouped_data['Shoes']
    initial_guess = [(max(shoes_data) - min(shoes_data)) / 2,  2 * np.pi / 4,
    0,  np.mean(shoes_data)]
    numeric_seasons = np.array(numeric_seasons) / max(numeric_seasons)
    popt, _ = curve_fit(sinusoid, numeric_seasons, shoes_data, p0=initial_guess,
bounds=([0, 0, -np.pi, min(shoes_data)], [np.inf, np.inf, np.pi, max(shoes_data)]))
    shoes_pred = sinusoid(np.array(numeric_seasons), *popt)

    plt.figure(figsize=(10, 6))
    plt.plot(numeric_seasons, pants_data, label='Pants (Actual)', marker='o', color='blue')
    plt.plot(numeric_seasons, pants_pred, label='Pants (Predicted)', linestyle='--', color='blue')
    plt.plot(numeric_seasons, shoes_data, label='Shoes (Actual)', marker='o', color='orange')
    plt.plot(numeric_seasons, shoes_pred, label='Shoes (Predicted)', linestyle='--', color='orange')

    plt.title('Regression Analysis: Pants and Shoes')
    plt.xlabel('Season (Numeric)')
    plt.ylabel('Number of Purchases')
    plt.xticks(ticks=numeric_seasons, labels=season_labels, rotation=45)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def skirt_sweater_regression(dataset, column_id: list[int]):
    '''
    This function performs a regression on the items skirt and sweater. Both items
    need sinusoid regression. We first start with a regression for one year.
    '''
    data_item = [entry[column_id[0]] for entry in dataset]
    data_season = [entry[column_id[1]] for entry in dataset]

    seasons = ['Spring', 'Summer', 'Fall', 'Winter']
    df = pd.DataFrame({'Item': data_item, 'Season': data_season})
    df['Season'] = pd.Categorical(df['Season'], categories=seasons, ordered=True)
    grouped_data = df.groupby(['Season', 'Item'],observed=False).size().unstack(fill_value=0)

    season_labels = grouped_data.index.str.replace('_2', '')
    numeric_seasons = list(range(1, len(season_labels) + 1))

    # for skirt
    skirt_data = grouped_data['Skirt']
    initial_guess = [(max(skirt_data) - min(skirt_data)) / 2,  2 * np.pi / 4,
    0,  np.mean(skirt_data)]
    numeric_seasons = np.array(numeric_seasons) / max(numeric_seasons)
    popt, _ = curve_fit(sinusoid, numeric_seasons, skirt_data, p0=initial_guess,
    bounds=([0, 0, -np.pi, min(skirt_data)], [np.inf, np.inf, np.pi, max(skirt_data)]))
    skirt_pred = sinusoid(np.array(numeric_seasons), *popt)

    # for sweater
    sweater_data = grouped_data['Sweater']
    initial_guess = [(max(sweater_data) - min(sweater_data)) / 2,  2 * np.pi / 4,
    0,  np.mean(sweater_data)]
    numeric_seasons = np.array(numeric_seasons) / max(numeric_seasons)
    popt, _ = curve_fit(sinusoid, numeric_seasons, sweater_data, p0=initial_guess,
bounds=([0, 0, -np.pi, min(sweater_data)], [np.inf, np.inf, np.pi, max(sweater_data)]))
    sweater_pred = sinusoid(np.array(numeric_seasons), *popt)

    plt.figure(figsize=(10, 6))
    plt.plot(numeric_seasons, skirt_data, label='Skirt (Actual)', marker='o', color='blue')
    plt.plot(numeric_seasons, skirt_pred, label='Skirt (Predicted)', linestyle='--', color='blue')
    plt.plot(numeric_seasons, sweater_data, label='Sweater (Actual)', marker='o', color='orange')
    plt.plot(numeric_seasons, sweater_pred, label='Sweater (Predicted)', linestyle='--', color='orange')

    plt.title('Regression Analysis: Skirt and Sweater')
    plt.xlabel('Season (Numeric)')
    plt.ylabel('Number of Purchases')
    plt.xticks(ticks=numeric_seasons, labels=season_labels, rotation=45)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def main():
    headers, dataset = read_csv("data/shopping_trends.csv")

    item_vs_season(dataset, [3,9])
    jacket_boots_regression(dataset, [3,9])
    hat_jeans_regression(dataset, [3,9])
    pants_shoes_regression(dataset, [3,9])
    skirt_sweater_regression(dataset, [3,9])





if __name__ == '__main__':
    main()
