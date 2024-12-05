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

# make the histogram data
new_headers = []
new_hist_dataset = []
for sub_dataset in sub_datasets:
    data = [entry[months_column] for entry in sub_dataset]
    hist, bins = np.histogram(data, bins=np.arange(14))
    new_headers.append(sub_dataset[0][category_column])
    new_hist_dataset.append(np.array([bins[1:-1], hist[1:]]).T)

interpolation_factor = 20
big_fig, figs = plt.subplots(4, 4)
fft_reconstructed = []
for x in range(4):
    for y in range(4):
        index = x+4*y
        bins = new_hist_dataset[index][:, 0]
        hist = new_hist_dataset[index][:, 1]
        # Here we perform a fourier analysis because in theory this datashould be cyclical with periods of a divisor of 12 months.
        fft_analysis = np.fft.fft(hist, 12)
        # Now we want to show what the fouier transform finds for the continuous function including for the next year.
        new_bins = np.linspace(1, 12, 12*interpolation_factor)
        # now we recontruct the values
        fft_reconstructed_values = np.fft.ifft(
            fft_analysis, len(new_bins))*interpolation_factor
        fft_reconstructed.append(fft_reconstructed_values)
        # now we plot the original data and the fourier in the same graph
        figs[x, y].plot(bins, hist, label="original data")
        figs[x, y].plot(new_bins, np.real(fft_reconstructed_values),
                        label="recontructed fft data", ls=':')
        # figs[x, y].plot(bins, np.real(fft_reconstructed_values),
        #                 label="recontructed fft data", ls=':')
        figs[x, y].set_title(new_headers[index])
        figs[x, y].set_xlabel("month")
        figs[x, y].set_ylabel("number of purchases")

big_fig.align_labels
plt.legend()
plt.show()
