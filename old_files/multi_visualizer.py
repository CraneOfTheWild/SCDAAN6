import data_visualizers as dv
import numpy as np
import matplotlib.pyplot as plt

# read in the data
headers, dataset = dv.read_csv("data/file.csv")
months_column = 18
category_column = 9
gender_column = 2


# show the general layout and the kind of plots we can use for fitting
# here we split by the category held in the category_column
_, sub_datasets = dv.split_on_column(headers, dataset, category_column)
# now we plot all these seperately (so with different scales!)
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
plt.tight_layout()
plt.show()

# now with the same scale
big_fig, figs = plt.subplots(4, 4)
for x in range(4):
    for y in range(4):
        index = x+4*y
        header, data = dv.get_column(
            headers, sub_datasets[index], months_column)
        hist, bins = np.histogram(data, bins=np.arange(14))
        hist = hist/np.mean(hist)
        figs[x, y].plot(bins[1:-1:], hist[1:])
        figs[x, y].set_title(sub_datasets[index][0][category_column])
        figs[x, y].set_xlabel("month")
        figs[x, y].set_ylabel("number of purchases")
big_fig.align_labels()
plt.tight_layout()
plt.show()


# show the assumption of one year being the same as the next when it seems to be true
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

# show where this assumption of one year being the same as the next when it seems to be false
category_to_show = 10
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


# now with gender
_, sub_datasets = dv.split_on_column(headers, dataset, gender_column)
big_fig, figs = plt.subplots(2)
for x in range(2):
    index = x
    header, data = dv.get_column(
        headers, sub_datasets[index], months_column)
    hist, bins = np.histogram(data, bins=np.arange(14))
    figs[x].plot(bins[1:-1:], hist[1:])
    figs[x].set_title(sub_datasets[index][0][gender_column])
    figs[x].set_xlabel("month")
    figs[x].set_ylabel("number of purchases")
big_fig.align_labels()
plt.tight_layout()
plt.show()
