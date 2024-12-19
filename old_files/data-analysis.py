
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
    pd_dataset = pd.read_csv(filename, low_memory=False)
    headers = pd_dataset.columns.to_list()
    dataset = pd_dataset.to_numpy()
    return headers, dataset

def plot_fourier(extended_grouped_data, group, seasons):
    n_components = 5
    fig, axes = plt.subplots(2, 3, figsize=(14, 10))
    axes = axes.flatten()

    print(group)
    for i, item in enumerate(group):
        ax = axes[i]
        y_data = extended_grouped_data[item].values
        reconstructed_data = fourier_fit(y_data, n_components)

        ax.plot(seasons, y_data, marker='o', label=f'{item} (Actual)')
        ax.plot(seasons, reconstructed_data, linestyle='--', label=f'{item} (FFT Reconstructed)')
        ax.set_title(f'Item: {item}')
        ax.set_xlabel('Season')
        ax.set_ylabel('Number of Purchases')
        plt.legend()
        ax.tick_params(axis='x', rotation=45)

    for j in range(len(group), len(axes)):
        fig.delaxes(axes[j])

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
    duplicated_grouped_data = grouped_data.copy()
    duplicated_grouped_data.index = duplicated_grouped_data.index.map(lambda x: f"{x}_2")
    extended_grouped_data = pd.concat([grouped_data, duplicated_grouped_data])
    extended_seasons = [season for season in extended_grouped_data.index]


    items = extended_grouped_data.columns.to_list()
    first_group, second_group, third_group, fourth_group = items[:6],items[6:12],items[12:18],items[18:24]

    plot_fourier(extended_grouped_data, first_group, extended_seasons)
    plot_fourier(extended_grouped_data, second_group, extended_seasons)
    plot_fourier(extended_grouped_data, third_group, extended_seasons)
    plot_fourier(extended_grouped_data, fourth_group, extended_seasons)

    plt.figure(figsize=(14, 8))
    sns.heatmap(grouped_data, annot=True, fmt="d", cmap='coolwarm', linewidths=0.5)
    plt.title('Heatmap: Frequency of Items Bought by Season')
    plt.xlabel('Item')
    plt.ylabel('Season')
    plt.xticks(rotation=45)
    plt.show()

def fourier_fit(data, n_components):
    fft_coeffs = np.fft.fft(data)  # Perform FFT
    fft_coeffs[n_components:] = 0  # Retain only the first n_components frequencies
    reconstructed = np.fft.ifft(fft_coeffs).real  # Perform inverse FFT
    return reconstructed

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

def polynomial_regression(x, y, degree):
    coeffs = np.polyfit(x, y, degree)
    poly_func = np.poly1d(coeffs)
    predictions = poly_func(x)
    return poly_func, predictions

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

def item_vs_date(dataset, column_id: list[int]):
    data_date = [entry[column_id[0]] for entry in dataset]
    data_item = [entry[column_id[1]] for entry in dataset]

    df = pd.DataFrame({'Date': data_date, 'Item': data_item})
    df['Date'] = pd.to_datetime(df['Date'])
    df['YearMonth'] = df['Date'].dt.to_period('M')

    item_purchases_over_time = (df.groupby(['YearMonth', 'Item']).size().reset_index(name='Total_Purchases'))
    pivot_data = item_purchases_over_time.pivot(index='YearMonth', columns='Item', values='Total_Purchases').fillna(0)


    items = pivot_data.columns.to_list()
    chunks = [items[:5], items[5:10], items[10:15], items[15:]]


    x_labels = pivot_data.index.to_timestamp().strftime('%Y-%m')
    numeric_time = range(len(x_labels))

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for i, chunk in enumerate(chunks):
        ax = axes[i]
        for item in chunk:
            ax.plot(numeric_time, pivot_data[item].values, marker='o', label=f'{item} (Actual)')
            if pivot_data[item].sum() > 0:
                poly_func, predictions = polynomial_regression(numeric_time, pivot_data[item].values, degree=2)
                ax.plot(numeric_time, predictions, linestyle='--', label=f'{item} (Predicted)')

        ax.set_title(f'Items {i * 5 + 1} to {i * 5 + len(chunk)}')
        ax.set_xlabel('Month')
        ax.set_ylabel('Number of Purchases')
        ax.set_xticks(numeric_time)
        ax.set_xticklabels(x_labels, rotation=45, ha='right')
        ax.legend(title='Item', bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.tight_layout()
    plt.show()

def plot_pak(pivot_data, group, numeric_time, x_labels):
    n_components = 5

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    for i, item in enumerate(group):
        ax = axes[i]
        y_data = pivot_data[item].values
        reconstructed_data = fourier_fit(y_data, n_components)

        ax.plot(numeric_time, y_data, marker='o', label=f'{item} (Actual)')
        ax.plot(numeric_time, reconstructed_data, linestyle='--', label=f'{item} (FFT Reconstructed)')

        ax.set_title(f'FFT for {item}')
        ax.set_xlabel('Month')
        ax.set_ylabel('Number of Purchases')
        ax.set_yscale('log')
        ax.set_xticks(numeric_time)
        ax.set_xticklabels(x_labels, rotation=45, ha='right')
        ax.legend()

    for j in range(len(group), len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()

def pak_item_vs_date(dataset, column_id: list[int]):
    data_date = [entry[column_id[0]] for entry in dataset]
    data_item = [entry[column_id[1]] for entry in dataset]

    df = pd.DataFrame({'Date': data_date, 'Item': data_item})
    df['Date'] = pd.to_datetime(df['Date'])
    df['YearMonth'] = df['Date'].dt.to_period('M')
    df.replace(r'\\N', pd.NA, inplace=True, regex=True)
    df = df.dropna()

    item_purchases_over_time = (df.groupby(['YearMonth', 'Item']).size().reset_index(name='Total_Purchases'))
    pivot_data = item_purchases_over_time.pivot(index='YearMonth', columns='Item', values='Total_Purchases').fillna(0)

    items = pivot_data.columns.to_list()
    x_labels = pivot_data.index.to_timestamp().strftime('%y-%m')
    numeric_time = range(len(x_labels))

    n_components = 5

    first_group = items[:6]
    second_group = items[6:12]
    third_group = items[12:]

    plot_pak(pivot_data, first_group, numeric_time, x_labels)

    plot_pak(pivot_data, second_group, numeric_time, x_labels)

    plot_pak(pivot_data, third_group, numeric_time, x_labels)



def main():
    # for the file shopping_trends.csv
    headers, dataset = read_csv("data/shopping_trends.csv")
    item_vs_season(dataset, [3,9])
    jacket_boots_regression(dataset, [3,9])
    hat_jeans_regression(dataset, [3,9])
    pants_shoes_regression(dataset, [3,9])
    skirt_sweater_regression(dataset, [3,9])

    # for the file file.csv
    headers2, dataset2 = read_csv("data/file.csv")
    item_vs_date(dataset2, [6,9])

    # for the file Pakistan_Ecommerce_Dataset.csv
    headers3, dataset3 = read_csv("data/Pakistan_Ecommerce_Dataset.csv")
    pak_item_vs_date(dataset3, [2,8])

if __name__ == '__main__':
    main()
