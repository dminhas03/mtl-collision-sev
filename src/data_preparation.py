from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt
import seaborn as sns
import osmnx as ox


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

    df['season'] = df['month'] % 12 // 3 + 1  # 1=winter, 2=spring, ...
    
    # drop the unnecessary date/time variables now so they don't cause issues when running model
    df = df.drop(columns=["DT_ACCDN", "HEURE_ACCDN", "JR_SEMN_ACCDN"], errors="ignore")

    return df




def missing_data(df):
    # For map later on: Get rid of missing coordinates and convert coordinates to num
    for coord in ["LOC_LAT", "LOC_LONG"]:
        if coord in df.columns:
            df.loc[:, coord] = pd.to_numeric(df[coord], errors="coerce")
    if {"LOC_LAT", "LOC_LONG"}.issubset(df.columns):
        df = df.dropna(subset=["LOC_LAT", "LOC_LONG"]).copy()

    # For columns consisting of counts, make sure there's a 0 if value dne
    count_cols = [c for c in df.columns if c.startswith("NB_")]
    for c in count_cols:
        if c in df.columns:
            df.loc[:, c] = df[c].fillna(0).astype(int)

    # use median for missing hour
    if "hour" in df.columns:
        med = df["hour"].median()
        fill_val = int(med) if pd.notna(med) else 0
        df.loc[:, "hour"] = df["hour"].fillna(fill_val).astype(int)

    # Day/Night flag
    df.loc[:, "is_night"] = ((df["hour"] < 6) | (df["hour"] > 20)).astype(int)

    # Categorical columns: fill with "N/A"
    for c in df.columns:
        if pd.api.types.is_object_dtype(df[c]) or pd.api.types.is_string_dtype(df[c]):
            df.loc[:, c] = df[c].astype("string").fillna("N/A")
    return df




def encode(df):
    cat_cols = df.select_dtypes(include=["object", "string", "category"]).columns.tolist()
    cat_cols = [c for c in cat_cols if c not in ["JR_SEMN_ACCDN", "HEURE_ACCDN"]]

    enc = OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore")
    enc_arr = enc.fit_transform(df[cat_cols])
    enc_df = pd.DataFrame(enc_arr, columns=enc.get_feature_names_out(cat_cols), index=df.index)

    df = pd.concat([df.drop(columns=cat_cols), enc_df], axis=1)
    return df





def main():
    # Load raw data into dataframe
    accidents = load_raw()
    
    # Columns we want to keep. 
    keep_cols = [
    # Time features
    'DT_ACCDN', 'JR_SEMN_ACCDN', "HEURE_ACCDN",

    # Location features
    'LOC_LAT', 'LOC_LONG', 'CD_CATEG_ROUTE', 'CD_SIT_PRTCE_ACCDN',

    # Environment and context
    'CD_COND_METEO', 'CD_ETAT_SURFC', 'CD_ECLRM', 'CD_ENVRN_ACCDN', 'CD_GENRE_ACCDN'
    ]
    accidents = accidents[keep_cols].copy()

    accidents = handle_time(accidents)
    accidents = missing_data(accidents)
    accidents = encode(accidents)

    print("\nAccidents: ", accidents.shape)
    print(accidents.head(5))


    ###################################################################
    #################### GENERATING NEGATIVE CASES ####################
    ###################################################################
    # We want ourmodel to predict accidents but the above dataset only contains accidents so we need to create no accident data
    # For coordinates we can use osmnx to get graph of Montreal's street network. 
    # From said graph we can get latitude and longitude of each node and from there we pick however many points we want.    
    # neg =  pd.DataFrame(columns=('LOC_LAT', 'LOC_LONG'))

    graph = ox.graph.graph_from_place("Montreal, Canada", network_type="drive")
    # fig, ax = ox.plot.plot_graph(graph)

    coord_list = []
    for node, data in graph.nodes(data=True):
        coord = {'LOC_LAT': data['y'], 'LOC_LONG': data['x']}
    #     latitude = data['y']  # Latitude
    #     longitude = data['x'] # Longitude
    #     print(f"Node {node}: Lat={latitude}, Lon={longitude}")
        coord_list.append(coord)
    neg =  pd.DataFrame(coord_list)
    print( "\n Coordinates: ", neg.shape)
    print(neg.head(5))




    # Save df
    save_processed(accidents)

if __name__ == "__main__":
    main()