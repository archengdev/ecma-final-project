import pandas as pd

file_path = "./angrist.dta"
data = pd.read_stata(file_path)
columns = ["v9", "v4", "v18", "v1", "v10", "v11", "v19"]
stats = data[columns].agg(['mean', 'std'])
print(stats)


file_path = "./basic/data.csv"
data = pd.read_csv(file_path)
columns = ["y1", "t1", "z1", "x1", "v10", "v11", "v19"]
stats = data[columns].agg(['mean', 'std'])
print(stats)
file_path = "./weak/data.csv"
data = pd.read_csv(file_path)
columns = ["y1", "t1", "z1", "x1", "v10", "v11", "v19"]
stats = data[columns].agg(['mean', 'std'])
print(stats)
file_path = "./endog/data.csv"
data = pd.read_csv(file_path)
columns = ["y1", "t1", "z1", "x1", "v10", "v11", "v19"]
stats = data[columns].agg(['mean', 'std'])
print(stats)
file_path = "./manyweak/data.csv"
data = pd.read_csv(file_path)
columns = ["y1", "t1", "z1", "x1", "v10", "v11", "v19"]
stats = data[columns].agg(['mean', 'std'])
print(stats)