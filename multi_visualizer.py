import data_visualizers as dv

headers, dataset = dv.read_csv("data/file.csv")

columns_to_test = [10, 11, 12]
dv.multi_comparison(headers, dataset, columns_to_test)
