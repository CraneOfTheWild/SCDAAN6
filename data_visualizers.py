# These scripts are made to visualize a dataset so as to understand its trends.

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from collections import Counter
from matplotlib.ticker import MaxNLocator

#######################
## DATA MANIPULATION ##
#######################


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


def get_column(headers: list, dataset, column_id=0) -> tuple[list, list]:
    """
    This function gets the header and data of a single column

    Keyword arguments:  
    |headers    -- an array of all the names of the columns
    |dataset    -- a dataset as a regular 2d numpy array
    |column_id  -- an integer number for the column starting at 0           (default 0)
    """
    data = np.array([entry[column_id] for entry in dataset])
    header = headers[column_id]
    return header, data


def get_columns_ratio(headers: list, dataset, column_id_0=0, column_id_1=1) -> tuple[list, list]:
    """
    This function makes a new data column by finding the entrywise ratio of two !NUMERICAL! columns.

    Keyword arguments:  
    |headers        -- an array of all the names of the columns
    |dataset        -- a dataset as a regular 2d numpy array
    |column_id_0    -- an integer number for the numerator column starting at 0     (default 0)
    |column_id_1    -- an integer number for the divisor column starting at 0       (default 1)
    """
    header_0, data_0 = get_column(headers, dataset, column_id_0)
    header_1, data_1 = get_column(headers, dataset, column_id_1)
    # TODO add checking if numerical data
    new_data = np.divide(data_0, data_1)
    new_header = f"ratio({header_0},{header_1})"
    return new_header, new_data


def get_self_fit_line(data: list, intercept=0, slope=1) -> list[list]:
    """
    Makes a 2d array with columns x,y following a line y = slope * x + intercept with x in the domain of data.

    Keyword arguments:  
    |data       -- a data column as a numpy array
    |intercept  -- a number to indicate y when x = 0            (default 0)
    |slope      -- a number to indicate the slope of the line   (default 1)
    """
    x_min = min(data)
    x_max = max(data)
    x = np.array([x_min, x_max])
    y = x * slope + intercept
    return np.array([x, y]).T


def get_self_fit_sinoid(data: list, amplitude=1, period=1, phase_shift=0, vertical_shift=0) -> list[list]:
    """
    Makes a 2d array with columns x,y following a sinoid:
        y = amplitude * sine( ((2 * pi) / period) * x + ((2 * pi) / period) * phase_shift) + vertical_shift 
    with x in the domain of data.

    Keyword arguments:  
    |data           -- a data column as a numpy array
    |amplitude      -- a number to indicate the distance between the mean and the max/min               (default 1)
    |period         -- a number to indicate the distance between one max and the next                   (default 1)
    |phase_shift    -- a number to indicate the place where the sine is at the mean and going to a max  (default 0)
    |vertical_shift -- a number to indicate the mean y of the sinoid                                    (default 0)
    """
    x_min = min(data)
    x_max = max(data)
    nr_periods = (x_max - x_min) / period
    x = np.linspace(x_min, x_max, nr_periods*20)
    period_proportion = (2 * np.pi) / period
    y = amplitude * np.sin(period_proportion * x +
                           period_proportion * phase_shift) + vertical_shift
    return np.array([x, y]).T


def split_on_column(headers: list, dataset, discrete_column=1) -> tuple[list, list[list[list]]]:
    """
    Split a data column by another columns discrete values.

    Keyword arguments:  
    |headers            -- a names array for each column of data
    |dataset            -- a dataset as a regular 2d numpy array
    |discrete_column    -- an integer number for the y column starting at 0     (default 0)
    """
    _, discrete_data = get_column(headers, dataset, discrete_column)
    discrete_data_set = np.unique(discrete_data)
    split_datasets = []
    for selector in discrete_data_set:
        new_dataset = np.array([entry for entry in dataset
                                if entry[discrete_column] == selector])
        split_datasets.append(new_dataset)
    return headers, np.array(split_datasets)


####################
## VISUALIZATIONS ##
####################


# Histogram

def histogram_graph_direct(figure, data: list) -> None:
    """
        This function makes a histogram plot of one column of the dataset.

        Keyword arguments:
        |figure     -- the figure in which to plot the histogram
        |data       -- a data column as a numpy array
    """
    figure.hist(data)


def histogram_graph(figure, dataset, column_id: int) -> None:
    """
        This function makes a histogram plot of one column of the dataset.

        Keyword arguments:  
        |figure     -- the figure in which to plot the histogram
        |dataset    -- a dataset as a regular 2d numpy array
        |column_id  -- an integer number for the column starting at 0
    """
    data = np.array([entry[column_id] for entry in dataset])
    return histogram_graph_direct(figure, data)


# Line graphs


def line_graph_direct(figure, x_data: list, y_data: list, x_header="x", y_header="y") -> None:
    """
    This function makes a line plot of two columns of the dataset.

    Keyword arguments:
    |x_data     -- a data column as a numpy array
    |y_data     -- a data column as a numpy array
    |x_header   -- a string discribing the data in the x_column     (deafault "x")
    |y_header   -- a string discribing the data in the y_column     (deafault "y")
    """
    xy_label = f"{x_header} vs {y_header}"
    # Here we sort the data based on data_0 which will go on the x-axis to give a nice line.
    sorted_indices = np.argsort(x_data)
    x_data = np.array(x_data)[sorted_indices]
    y_data = np.array(y_data)[sorted_indices]
    # Plotting.
    figure.plot(x_data, y_data, label=xy_label)


def line_graph(figure, headers: list, dataset, x_column=0, y_column=1) -> None:
    """
    This function makes a line plot of two columns of the dataset.

    Keyword arguments:  
    |headers    -- a names array for each column of data
    |dataset    -- a dataset as a regular 2d numpy array
    |x_column   -- an integer number for the x column starting at 0         (default 0)
    |y_column   -- an integer number for the y column starting at 0         (default 1)
    |title      -- the title of the plot                                    (default "line plot")
    """
    x_header, x_data = get_column(headers, dataset, x_column)
    y_header, y_data = get_column(headers, dataset, y_column)
    return line_graph_direct(figure, x_data, y_data, x_header, y_header)


# Scatter graphs


def scatter_graph_direct(figure, x_data: list, y_data: list, x_header="x", y_header="y") -> None:
    """
    This function makes a scatter plot of two columns of the dataset.

    Keyword arguments:
    |x_data     -- a data column as a numpy array
    |y_data     -- a data column as a numpy array
    |x_header   -- a string discribing the data in the x_column     (deafault "x")
    |y_header   -- a string discribing the data in the y_column     (deafault "y")
    """
    xy_label = f"x-axis: {x_header} vs y-axis: {y_header}"
    # Plotting.
    figure.scatter(x_data, y_data, alpha=0.3, label=xy_label)


def scatter_graph(figure, headers: list, dataset, x_column, y_column) -> None:
    """
    This function makes a line plot of two columns of the dataset.

    Keyword arguments:  
    |headers    -- a names array for each column of data
    |dataset    -- a dataset as a regular 2d numpy array
    |x_column   -- an integer number for the x column starting at 0         (default 0)
    |y_column   -- an integer number for the y column starting at 0         (default 1)
    """
    x_header, x_data = get_column(headers, dataset, x_column)
    y_header, y_data = get_column(headers, dataset, y_column)
    return scatter_graph_direct(figure, x_data, y_data, x_header, y_header)


def multi_comparison(headers: list, dataset, columns=[0]) -> None:
    """
    Makes a comparison of all columns specified in scatter graphs.
    It does this by making subplots for all combinations.
    On the diagonal where a column is compared to itself a histogram is used.

    Keyword arguments:  
    |headers    -- a names array for each column of data
    |dataset    -- a dataset as a regular 2d numpy array
    |columns    -- an integer number array for the columns to be compared (default [0])
    """
    n = len(columns)
    _, figs = plt.subplots(n, n)
    for x in range(n):
        for y in range(n):
            if x == y:
                figs[x, y].set_title(headers[columns[x]])
                histogram_graph(figs[x, y], dataset, columns[x])
            else:
                figs[x, y].set_title(headers[columns[x]] +
                                     " vs " +
                                     headers[columns[y]])
                scatter_graph(figs[x, y], headers, dataset,
                              columns[x], columns[y])
    plt.show()


#################
## LEGACY CODE ##
#################


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
