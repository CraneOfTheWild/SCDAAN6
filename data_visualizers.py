# These scripts are made to visualize a dataset so as to understand its trends.

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from collections import Counter


def read_csv(filename: str, header_length=1) -> tuple[list, list[list]]:
    """
    Read in a csv file into an array.

    Keyword arguments:  
    |filename   -- the name of the csv file including the .csv
    """
    pd_dataset = pd.read_csv(filename)
    headers = pd_dataset.columns.to_list()
    dataset = pd_dataset.to_numpy()
    return headers, dataset


def histogram_comparison(dataset,column_id: int, title="histogram", use_bins=False):
    """
        This function makes a histogram plot of one column of the dataset.

        Keyword arguments:  
        |dataset    -- a dataset as a regular array of values
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


def main():
    headers, dataset = read_csv("data/Marine_Fish_Data.csv", 1)
    histogram_comparison(dataset, 0, title=headers[0])
    histogram_comparison(dataset, 5, title=headers[5], use_bins=True)


if __name__ == "__main__":
    main()
