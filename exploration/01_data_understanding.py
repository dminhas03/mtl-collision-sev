import pandas as pd
import os
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))
from src.utils_io import load_raw, save_filtered 

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

# Columns we want to keep:
keep_cols = [
    # Target variable
    'GRAVITE',
    # Casualty counts
    'NB_MORTS', 'NB_BLESSES_GRAVES', 'NB_BLESSES_LEGERS',
    'NB_VICTIMES_TOTAL',
    'NB_DECES_PIETON', 'NB_BLESSES_PIETON', 'NB_VICTIMES_PIETON',
    'NB_DECES_MOTO', 'NB_BLESSES_MOTO', 'NB_VICTIMES_MOTO',
    'NB_DECES_VELO', 'NB_BLESSES_VELO', 'NB_VICTIMES_VELO',

    # Time features
    'DT_ACCDN', 'JR_SEMN_ACCDN', "HEURE_ACCDN",

    # Location features
    'LOC_LAT', 'LOC_LONG', 'CD_CATEG_ROUTE', 'CD_SIT_PRTCE_ACCDN',

    # Environment and context
    'CD_COND_METEO', 'CD_ETAT_SURFC', 'CD_ECLRM', 'CD_ENVRN_ACCDN', 'CD_GENRE_ACCDN'
]

df = df[keep_cols].copy()
save_filtered(df)
