import pandas as pd
import os

script_dir = os.path.dirname(__file__)
data_path = os.path.join(script_dir, '..', 'data', 'collisions_routieres.csv')

# Loading csv file
df = pd.read_csv(data_path)

print(df.head())

print("\nShape (rows, columns): ", df.shape)

pd.set_option('display.max_rows', None)
print("\nData types:\n", df.dtypes)
pd.reset_option('display.max_rows') 

print("\nCheck for missing values: \n",df.isnull().sum())

print("\n Descriptive Statistics:\n", df.describe())

missing_percent = df.isnull().mean() * 100
print("\nPercentage of missing values per column:\n", missing_percent)