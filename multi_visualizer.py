import data_visualizers as dv
import numpy as np
import matplotlib.pyplot as plt

# read in the data
headers, dataset = dv.read_csv("data/file.csv")
months_column = 18
category_column = 9
_, sub_datasets = dv.split_on_column(headers, dataset, category_column)

# # show the general layout and the kind of plots we can use
big_fig, figs = plt.subplots(4, 4)
for x in range(4):
    for y in range(4):
        index = x+4*y
        header, data = dv.get_column(
            headers, sub_datasets[index], months_column)
        hist, bins = np.histogram(data, bins=np.arange(14))
        figs[x, y].plot(bins[1:-1:], hist[1:])
        figs[x, y].set_title(sub_datasets[index][0][category_column])
        figs[x, y].set_xlabel("month")
        figs[x, y].set_ylabel("number of purchases")
big_fig.align_labels()
plt.show()

# show the assumption of one year being the same as the next
category_to_show = 13
header_column, data_column = dv.get_column(headers,
                                           sub_datasets[category_to_show],
                                           months_column)
multi_year_sub_data = data_column+12
multi_year_sub_data = np.hstack((multi_year_sub_data,
                                 data_column))
hist, bins = np.histogram(multi_year_sub_data, bins=np.arange(26))
plt.plot(bins[1:-1], hist[1:])
plt.title(sub_datasets[category_to_show][0][category_column])
plt.xlabel("month")
plt.ylabel("number of purchases")
plt.show()
