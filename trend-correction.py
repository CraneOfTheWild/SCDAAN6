'''
This code combines the codes 'data-analysis.py' and 'fourier_fit.py' to perform
a trend correction on the file.csv file.
'''

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from collections import Counter
import seaborn as sns
from scipy.optimize import curve_fit

FILENAME = "data/file.csv"
DATE_COLUMN_NAME = "Transaction_Date"
CATEGORY_COLUMN_NAME = "Product_Category"
FOURIER_COMPONENTS = 30

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

def item_vs_date(dataset, column_id: list[int]):
    data_date = [entry[column_id[0]] for entry in dataset]
    data_item = [entry[column_id[1]] for entry in dataset]

    df = pd.DataFrame({'Date': data_date, 'Item': data_item})
    df['Date'] = pd.to_datetime(df['Date'])
    df['Day'] = df['Date'].dt.to_period('D')

    item_purchases_over_time = (df.groupby(['Day', 'Item']).size().reset_index(name='Total_Purchases'))
    pivot_data = item_purchases_over_time.pivot(index='Day', columns='Item', values='Total_Purchases').fillna(0)

    items = pivot_data.columns.to_list()
    chunks = [items[:5], items[5:10], items[10:15], items[15:]]

    x_labels = pivot_data.index.to_timestamp().strftime('%Y-%m')
    numeric_time = range(len(x_labels))

    data_predicted = {}
    for i, chunk in enumerate(chunks):
        for item in chunk:
            if pivot_data[item].sum() > 0:
                poly_func, predictions = polynomial_regression(numeric_time, pivot_data[item].values, degree=2)
                data_predicted[item]= predictions

    return data_predicted

def rolling_average(nr_days):
    # read in the data
    df = pd.read_csv(FILENAME, parse_dates=[DATE_COLUMN_NAME])
    # aggregate by date
    df_per_day = pd.crosstab(df[DATE_COLUMN_NAME].dt.date,
                             df[CATEGORY_COLUMN_NAME])
    # sort for certainty
    df_per_day = df_per_day.sort_index()
    # get roling average
    df_rolling_avg = df_per_day.rolling(window=nr_days,
                                        center=True,
                                        min_periods=1).mean()
    df_rolling_avg = df_rolling_avg.dropna()
    # plotting
    df_rolling_avg.plot.line(ax=plt.gca())
    plt.xlabel("Date")
    plt.ylabel("Sales_Number")
    plt.title(f"roling average of sales for surrounding {nr_days} days")
    plt.show()

def fourier_fit(nr_days=1):
    # read in the data
    df = pd.read_csv(FILENAME, parse_dates=[DATE_COLUMN_NAME])
    # aggregate by date
    df_per_day = pd.crosstab(df[DATE_COLUMN_NAME].dt.date,
                             df[CATEGORY_COLUMN_NAME])
    # sort for certainty
    df_per_day = df_per_day.sort_index()
    # get roling average
    df_rolling_avg = df_per_day.rolling(window=nr_days,
                                        center=True,
                                        min_periods=1).mean()
    df_rolling_avg = df_rolling_avg.dropna()
    # convert to numpy for use of np.fft
    column_names = df_rolling_avg.columns.tolist()
    real_data = df_rolling_avg.to_numpy()

    # save the fourier analysis values
    fourier_values = []
    real_data_dict = {}
    for x in range(5):
        for y in range(4):
            index = x + 4 * y
            # make a fourier fit

            real_data_dict[column_names[index]] = real_data.T[index]
            fft_analysis = np.fft.fft(real_data.T[index])
            fft_analysis_truncated = np.zeros_like(fft_analysis)
            fft_analysis_truncated[:FOURIER_COMPONENTS] = fft_analysis[:FOURIER_COMPONENTS]
            fourier_values.append(fft_analysis[:FOURIER_COMPONENTS])
            reconstructed_data = np.fft.ifft(fft_analysis_truncated).real

    return real_data_dict

def trend_correction(real_data, fitted_data, nr_days=1):

    corrected_data = [real_data[key] - fitted_data[key] for key, value in real_data.items()]

    df = pd.read_csv(FILENAME, parse_dates=[DATE_COLUMN_NAME])
    # aggregate by date
    df_per_day = pd.crosstab(df[DATE_COLUMN_NAME].dt.date,
                             df[CATEGORY_COLUMN_NAME])
    # sort for certainty
    df_per_day = df_per_day.sort_index()
    # get roling average
    df_rolling_avg = df_per_day.rolling(window=nr_days,
                                        center=True,
                                        min_periods=1).mean()
    df_rolling_avg = df_rolling_avg.dropna()
    column_names = df_rolling_avg.columns.tolist()
    big_fig, figs = plt.subplots(5, 4, figsize=(20,16))

    fourier_values = []
    for x in range(5):
        for y in range(4):
            index = x + 4 * y
            figs[x, y].plot(df_rolling_avg.index, corrected_data[index],
                            label="Trend Corrected Data")
            figs[x, y].set_ylabel("Sales Number Corrected")
            figs[x, y].set_title(column_names[index])
    big_fig.align_labels()
    plt.tight_layout()
    big_fig.subplots_adjust(top=0.88, wspace=0.4, hspace=0.6)
    plt.legend()
    plt.show()
    return fourier_values

def main():

    headers2, dataset2 = read_csv("data/file.csv")
    fitted_data = item_vs_date(dataset2, [6,9])
    real_data = fourier_fit(14)
    trend_correction(real_data, fitted_data, 14)



if __name__ == '__main__':
    main()
