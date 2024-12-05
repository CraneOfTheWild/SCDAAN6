import data_visualizers as dv
import numpy as np
import matplotlib.pyplot as plt

# read in the data
headers, dataset = dv.read_csv("data/file.csv")
months_column = 18
category_column = 9


# show the general layout and the kind of plots we can use for fitting
# here we split by the category held in the category_column
_, sub_datasets = dv.split_on_column(headers, dataset, category_column)

new_hist_dataset = []
for sub_dataset in sub_datasets:
    data = [entry[months_column] for entry in sub_dataset]
    hist, bins = np.histogram(data, bins=np.arange(14))
    hist = hist[1:]
    normalized_hist = np.array(hist) / np.mean(hist)
    new_hist_dataset.append((sub_dataset[0][category_column], normalized_hist))

headers = [headers[category_column], "months-histogram"]

# we now have a new dataset with headers and new_hist dataset
