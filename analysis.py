import data_visualizers as dv

headers, dataset = dv.read_csv('ecommerce_customer_data_large.csv')
dv.scatter_plot_comparison(dataset, x_column=0, y_column=1)
