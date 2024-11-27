# These scripts are made to visualize a dataset so as to understand its trends.

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
    if use_bins:
        plt.title(title)
        plt.hist(data)
    else: # taken from https://stackoverflow.com/questions/28418988/how-to-make-a-histogram-from-a-list-of-strings
        counts = Counter(data)
        df = pd.DataFrame.from_dict(counts, orient='index')
        df.plot(kind='bar', title=title)
    plt.show()


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
    plt.scatter(x_data,y_data)
    plt.show()


def coloured_scatter_plot_comparison(dataset, x_column=0, y_column=1, colour_column=2, title="scatter plot", xlabel="x", ylabel="y", colour_label="") -> None:
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
    """
    colour_data = [entry[colour_column] for entry in dataset]
    colour_data_set = list(set(colour_data))
    data = [(entry[x_column], entry[y_column], entry[colour_column]) for entry in dataset]
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    for colour in colour_data_set:
        sub_data = np.array([entry[:-1] for entry in data if entry[2] == colour])
        plt.scatter(sub_data.T[0], sub_data.T[1], label=colour + " " + colour_label)
        plt.legend()
    plt.show()


def main():
    headers, dataset = read_csv("data/Marine_Fish_Data.csv")
    # histogram_comparison(dataset, 0, title=headers[0])
    # histogram_comparison(dataset, 5, title=headers[5], use_bins=True)
    
    # x_column = 5
    # y_column = 4
    # scatter_plot_comparison(dataset, x_column, y_column,
    #                         title="fish population for average size of the fish",
    #                         xlabel=headers[x_column], ylabel=headers[y_column])

    x_column = 5
    y_column = 4
    colour_column = 8
    coloured_scatter_plot_comparison(dataset, x_column, y_column, colour_column,
                                     title="fish population for average size of the fish",
                                     xlabel=headers[x_column], ylabel=headers[y_column],
                                     colour_label=headers[colour_column])


if __name__ == "__main__":
    main()
