# These scripts are made to visualize a dataset so as to understand its trends.

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from collections import Counter
from matplotlib.ticker import MaxNLocator


def read_csv(filename: str, drop_nan=False) -> tuple[list, list[list]]:
    """
    Read in a csv file into an array.

    Keyword arguments:  
    |filename   -- the name of the csv file including the .csv
    |drop_nan   -- if nan values should be dropped
    """
    pd_dataset = pd.read_csv(filename)
    headers = pd_dataset.columns.to_list()
    pd_dataset = pd_dataset.dropna()
    dataset = pd_dataset.to_numpy()
    return headers, dataset


def histogram_comparison_direct(data, title="histogram", use_bins=False) -> None:
    """
        This function makes a histogram plot of one column of the dataset.

        Keyword arguments:  
        |data       -- a data column as a numpy array
        |title      -- the title of the plot                                    (default "histogram")
        |use_bins   -- a boolean to signify if bin aggregation can be used      (default False)
    """
    if use_bins:
        plt.title(title)
        plt.hist(data)
    else:  # taken from https://stackoverflow.com/questions/28418988/how-to-make-a-histogram-from-a-list-of-strings
        counts = Counter(data)
        df = pd.DataFrame.from_dict(counts, orient='index')
        df.plot(kind='bar', title=title)
    plt.show()


def histogram_comparison(dataset, column_id: int, title="histogram", use_bins=False) -> None:
    """
        This function makes a histogram plot of one column of the dataset.

        Keyword arguments:  
        |dataset    -- a dataset as a regular 2d numpy array
        |column_id  -- an integer number for the column starting at 0
        |title      -- the title of the plot                                    (default "histogram")
        |use_bins   -- a boolean to signify if bin aggregation can be used      (default False)
    """
    data = [entry[column_id] for entry in dataset]
    histogram_comparison_direct(data, title, use_bins)


def scatter_plot_comparison(dataset, x_column=0, y_column=1, title="scatter plot", x_label="x", y_label="y") -> None:
    """
        This function makes a scatter plot of two columns of the dataset.

        Keyword arguments:  
        |dataset    -- a dataset as a regular 2d numpy array
        |x_column   -- an integer number for the x column starting at 0         (default 0)
        |y_column   -- an integer number for the y column starting at 0         (default 1)
        |title      -- the title of the plot                                    (default "histogram")
        |x_label    -- the label for the x-axis                                 (default "x")
        |y_label    -- the label for the y-axis                                 (default "y")
    """
    x_data = [entry[x_column] for entry in dataset]
    y_data = [entry[y_column] for entry in dataset]
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.scatter(x_data, y_data, alpha=0.3)
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True, prune='lower', nbins=6))
    plt.gca().yaxis.set_major_locator(MaxNLocator(integer=True, prune='both', nbins=6))
    plt.show()


def coloured_scatter_plot_comparison(dataset, x_column=0, y_column=1, colour_column=2, title="scatter plot", x_label="x", y_label="y", colour_label="", show_legend=True) -> None:
    """
        This function makes a scatter plot of two columns of the dataset.

        Keyword arguments:  
        |dataset        -- a dataset as a regular 2d numpy array
        |x_column       -- an integer number for the x column starting at 0         (default 0)
        |y_column       -- an integer number for the y column starting at 0         (default 1)
        |colour_column  -- an integer number for the colour column starting at 0    (default 2)
        |title          -- the title of the plot                                    (default "histogram")
        |x_label        -- the label for the x-axis                                 (default "x")
        |y_label        -- the label for the y-axis                                 (default "y")
        |colour_label   -- the label for the colours                                (default "")
        |show_legend    -- a boolean for if the legend is shown                     (default True)
    """
    colour_data = [entry[colour_column] for entry in dataset]
    colour_data_set = list(set(colour_data))
    data = [(entry[x_column], entry[y_column], entry[colour_column])
            for entry in dataset]
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    for colour in colour_data_set:
        sub_data = np.array([entry[:-1]
                            for entry in data if entry[2] == colour])
        plt.scatter(sub_data.T[0], sub_data.T[1],
                    label=colour + " " + colour_label, alpha=0.3)
        plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True, prune='lower', nbins=6))
        plt.gca().yaxis.set_major_locator(MaxNLocator(integer=True, prune='both', nbins=6))
        if show_legend:
            plt.legend()
    plt.show()


def main():
    headers, dataset = read_csv("data/file.csv")
    # these are the rows with numbers ie integers or floats
    numerical_rows = [0, 1, 4, 5, 10, 11, 12, 15, 16, 17, 19]
    # these are the rows with discrete values ie strings or integers with limit range
    discrete_rows = [1, 2, 3, 6, 7, 9, 13, 14, 17, 18, 19]

    # here are some examples for how to use these visualizers
    # in your own code use:
    #               import data_visualizers as dv
    # after that you can call any function here as dv.function_name()

    # histograms to show the distribution of that column set use_bins to true for numerical data
    discrete_row = 2
    histogram_comparison(dataset, discrete_row, title=headers[discrete_row])

    # numerical_row = 11
    # histogram_comparison(dataset, numerical_row,
    #                      title=headers[numerical_row], use_bins=True)

    # both these columns should hold numeric data
    x_column = 11
    y_column = 17
    scatter_plot_comparison(dataset, x_column, y_column,
                            title="scatter plot of two variables",
                            x_label=headers[x_column], y_label=headers[y_column])

    # both these columns should hold numeric data
    x_column = 11
    y_column = 17
    # this column should hold discrete data
    colour_column = 2
    coloured_scatter_plot_comparison(dataset, x_column, y_column, colour_column,
                                     x_label=headers[x_column], y_label=headers[y_column],
                                     colour_label=headers[colour_column])


if __name__ == "__main__":
    main()
