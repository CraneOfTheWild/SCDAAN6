import data_visualizers as dv
import numpy as np
import matplotlib.pyplot as plt

# read in the data
headers, dataset = dv.read_csv("data/file.csv")
months_column = 18
category_column = 9
_, sub_datasets = dv.split_on_column(headers, dataset, category_column)

# show the general layout
big_fig, figs = plt.subplots(4, 4)
for x in range(4):
    for y in range(4):
        index = x+4*y
        dv.histogram_graph(figs[x, y], sub_datasets[index], 18)
        figs[x, y].set_title(sub_datasets[index][0][9])
        figs[x, y].set_xlabel("month")
        figs[x, y].set_ylabel("number of purchases")
big_fig.align_labels()
plt.show()

# show the assumption of one year being the same as the next
