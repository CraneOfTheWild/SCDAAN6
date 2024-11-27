# These scripts are made to visualize a dataset so as to understand its trends.

import numpy as np
import matplotlib.pyplot as plt
import csv


def read_csv(filename: str) -> list[list]:
    """
    Read in a csv file into an array.
    https://www.geeksforgeeks.org/reading-csv-files-in-python/

    Keyword arguments:  
    |filename   -- the name of the csv file including the .csv
    """
    my_data = np.genfromtxt(filename, delimiter=',')
    return my_data


def histogram_comparison(dataset,column_id: int, title="histogram", use_bins=False, nr_bins=3):
    """
        This function makes a histogram plot of one column of the dataset.

        Keyword arguments:  
        |dataset    -- a dataset as a regular array of values
        |column_id  -- an integer number for the column starting at 0
        |title      -- the title of the plot                                    (default "histogram")
        |use_bins   -- a boolean to signify if bin agregation should be used    (default False)
        |nr_bins    -- an integer for the ammount of bins to be used            (default 3)
    """
    data = np.array(dataset).T[column_id]
    plt.title(title)
    plt.hist(data)
    plt.show()


def main():
    dataset = read_csv("data/Marine_Fish_Data.csv")
    histogram_comparison(dataset, 5, "average size")


if __name__ == "__main__":
    main()
