from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder


import sys
if __name__ == "__main__" and (Path.cwd() / "src").exists():
    # make sure project root is on sys.path when run as a script
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.utils_io import load_raw, save_processed

def handle_time(df):
    # Extract year, month, and day from date of accident
    df["DT_ACCDN"] = pd.to_datetime(df["DT_ACCDN"], errors="coerce")
    df["year"] = df["DT_ACCDN"].dt.year
    df["month"] = df["DT_ACCDN"].dt.month
    df["day"] = df["DT_ACCDN"].dt.day

    # Get weekday
    day_map = {
    "LU": 0,  # Lundi - Monday
    "MA": 1,  # Mardi - Tuesday
    "ME": 2,  # Mercredi - Wednesday
    "JE": 3,  # Jeudi - Thursday
    "VE": 4,  # Vendredi - Friday
    "SA": 5,  # Samedi - Saturday
    "DI": 6   # Dimanche - Sunday
    }
    
    # Formatting hour variable (makeing it datetime)
    start_times = df["HEURE_ACCDN"].astype("string").str.split("-").str[0]
    df["hour"] = pd.to_datetime(start_times, format="%H:%M:%S", errors="coerce").dt.hour

    # Creating weekend flag to identify if a day is a weekend
    df["weekday"] = df["JR_SEMN_ACCDN"].map(day_map)
    df["is_weekend"] = (df["weekday"] >= 5).astype(int)

    # Rush hour flag
    df["is_rush_hour"] = df["hour"].between(7, 9) | df["hour"].between(15, 18)
    df["is_rush_hour"] = df["is_rush_hour"].fillna(False).astype(int)
    
    # drop the unnecessary date/time variables now so they don't cause issues when running model
    df = df.drop(columns=["DT_ACCDN", "HEURE_ACCDN", "JR_SEMN_ACCDN"], errors="ignore")

    return df


def missing_data(df):

    # Get rid of any records missing GRAVITE
    df = df[df["GRAVITE"].notna()].copy()

    # For map later on: Get rid of missing coordinates and convert coordinates to num
    for coord in ["LOC_LAT", "LOC_LONG"]:
        if coord in df.columns:
            df[coord] = pd.to_numeric(df[coord], errors="coerce")
    if {"LOC_LAT", "LOC_LONG"}.issubset(df.columns):
        df = df.dropna(subset=["LOC_LAT", "LOC_LONG"])

    # For columns consisting of counts, make sure there's a 0 if value dne
    count_cols = [c for c in df.columns if c.startswith("NB_")]
    for c in count_cols:
        if c in df.columns:
            df[c] = df[c].fillna(0).astype(int)

    # Set missing hour to median
    if "hour" in df.columns:
        med = df["hour"].median()
        if pd.notna(med):
            df["hour"] = df["hour"].fillna(med).astype(int)
        else:
            df["hour"] = df["hour"].fillna(0).astype(int)  # If median doesn't exist, set 0

    # Categorical columns: fill with "N/A"
    cat_candidates = [
        "CD_CATEG_ROUTE", "CD_SIT_PRTCE_ACCDN",
        "CD_COND_METEO", "CD_ETAT_SURFC", "CD_ECLRM",
        "CD_ENVRN_ACCDN", "CD_GENRE_ACCDN", "JR_SEMN_ACCDN"
    ]
    cat_cols = [c for c in cat_candidates if c in df.columns]
    for c in cat_cols:
        df[c] = df[c].astype("string").fillna("N/A")

    return df

def encode(df):
    cat_cols = df.select_dtypes(include=["object", "string", "category"]).columns.tolist()
    cat_cols = [c for c in cat_cols if c not in ["GRAVITE", "JR_SEMN_ACCDN", "HEURE_ACCDN"]]

    enc = OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore")
    enc_arr = enc.fit_transform(df[cat_cols])
    enc_df = pd.DataFrame(enc_arr, columns=enc.get_feature_names_out(cat_cols), index=df.index)

    df = pd.concat([df.drop(columns=cat_cols), enc_df], axis=1)
    return df


def main():
    # Load raw data into dataframe
    df = load_raw()
    
    # Columns we want to keep:
    keep_cols = [
    # Target variable
    'GRAVITE',
    # Casualty counts
    # 'NB_MORTS', 'NB_BLESSES_GRAVES', 'NB_BLESSES_LEGERS',
    # 'NB_VICTIMES_TOTAL',
    # 'NB_DECES_PIETON', 'NB_BLESSES_PIETON', 'NB_VICTIMES_PIETON',
    # 'NB_DECES_MOTO', 'NB_BLESSES_MOTO', 'NB_VICTIMES_MOTO',
    # 'NB_DECES_VELO', 'NB_BLESSES_VELO', 'NB_VICTIMES_VELO',

    # Time features
    'DT_ACCDN', 'JR_SEMN_ACCDN', "HEURE_ACCDN",

    # Location features
    'LOC_LAT', 'LOC_LONG', 'CD_CATEG_ROUTE', 'CD_SIT_PRTCE_ACCDN',

    # Environment and context
    'CD_COND_METEO', 'CD_ETAT_SURFC', 'CD_ECLRM', 'CD_ENVRN_ACCDN', 'CD_GENRE_ACCDN'
    ]
    df = df[keep_cols].copy()

    # Apply methods
    df = handle_time(df)
    df = missing_data(df)
    df = encode(df)

    # Save df
    save_processed(df)

    # print(df.shape)

    # check variables types
    # pd.set_option('display.max_rows', None)
    # print("\nData types:\n", df.dtypes)
    # pd.reset_option('display.max_rows') 

if __name__ == "__main__":
    main()