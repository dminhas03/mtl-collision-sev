import sys
import osmnx as ox
import numpy as np
import pandas as pd
import geopandas as gpd
from pathlib import Path
from shapely.geometry import LineString
from sklearn.preprocessing import OneHotEncoder


if __name__ == "__main__" and (Path.cwd() / "src").exists():
    # make sure project root is on sys.path when run as a script
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.utils_io import load_raw, save_processed



def handle_time(df):
    """
    Takes as input dataframe and creates time/date variables
    """
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
    
    # drop the unnecessary date/time variables now 
    df = df.drop(columns=["DT_ACCDN", "HEURE_ACCDN", "JR_SEMN_ACCDN"], errors="ignore")

    return df



def missing_data(df):
    """
    Takes as input dataframe and handles missing data
    """

    # coordinates
    for coord in ["LOC_LAT", "LOC_LONG"]:
        if coord in df.columns:
            df[coord] = pd.to_numeric(df[coord], errors="coerce")
    if {"LOC_LAT", "LOC_LONG"}.issubset(df.columns):
        df = df.dropna(subset=["LOC_LAT", "LOC_LONG"]).copy()

    # for all count variables, var starting with NB_, we fill in missing data as 0
    count_cols = [c for c in df.columns if c.startswith("NB_")]
    for c in count_cols:
        df[c] = df[c].fillna(0).astype(int)

    # If hour missing then fill in median
    if "hour" in df.columns:
        med = df["hour"].median()
        df["hour"] = df["hour"].fillna(int(med) if pd.notna(med) else 0).astype(int)

    # create is_night flag based on hour
    df["is_night"] = ((df["hour"] < 6) | (df["hour"] > 20)).astype(int)

    # for missing data in coded categoricals (CD_*) we set to string with "N/A" instead of NaNs
    code_cols = [
        "CD_CATEG_ROUTE", "CD_SIT_PRTCE_ACCDN",
        "CD_COND_METEO", "CD_ETAT_SURFC", "CD_ECLRM", "CD_ENVRN_ACCDN"
    ]
    for c in code_cols:
        if c in df.columns:
            s = pd.to_numeric(df[c], errors="coerce").round().astype("Int64")
            s = s.astype("string").fillna("N/A")
            df[c] = s  

    # Safety against NaNs in fields from bad/missing dates 
    for c, fallback in [
        ("weekday", df["weekday"].mode().iloc[0] if "weekday" in df and not df["weekday"].mode().empty else 0),
        ("month",   df["month"].mode().iloc[0]   if "month"   in df and not df["month"].mode().empty   else 1),
        ("year",    int(df["year"].median())     if "year"    in df and pd.notna(df["year"].median())  else 2019),
        ("day",     15),
    ]:
        if c in df.columns:
            df[c] = df[c].fillna(fallback).astype(int, errors="ignore")

    return df


def encode(df):
    """
    Takes as input dataframe and one-hot-encodes categorical variables
    """
    cat_cols = df.select_dtypes(include=["object", "string", "category"]).columns.tolist()
    cat_cols = [c for c in cat_cols if c not in ["JR_SEMN_ACCDN", "HEURE_ACCDN", "is_accident"]]

    enc = OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore")
    enc_arr = enc.fit_transform(df[cat_cols])
    enc_df = pd.DataFrame(enc_arr, columns=enc.get_feature_names_out(cat_cols), index=df.index)

    df = pd.concat([df.drop(columns=cat_cols), enc_df], axis=1)
    return df


###################################################################
#################### GENERATING NEGATIVE CASES ####################
###################################################################

def get_road_points(spacing_m=100,location="Montréal, Quebec, Canada", max_points=None):
    """
    Parameters:
        - spacing_m : spacing between points simulated on each street
        - location  : location of street network
        - max_points: max # of points to collect

    Use OSMNX to get graph of Montreal's street network.
    Use edges between graph nodes to simulate street coordinates for our negative case.
    """
    graph = ox.graph.graph_from_place(location, network_type="drive")
    graph_m = ox.project_graph(graph)
    nodes, edges = ox.graph_to_gdfs(graph_m)

    pts = []
    for _, row in edges.iterrows():
        geom = row.get("geometry", None)
        if geom is None:
            continue
        if not isinstance(geom, LineString):
            continue
        
        length = geom.length  # in meters 
        # include both endpoints (i=0) and last step
        n_steps = int(length // spacing_m)
        for i in range(n_steps + 1):
            d = min(i * spacing_m, length)
            p = geom.interpolate(d)
            pts.append(p)
            if max_points is not None and len(pts) >= max_points:
                break
        if max_points is not None and len(pts) >= max_points:
            break

    # Build GeoDataFrame in projected CRS, then convert to WGS84 (lat/lon)
    gdf_pts = gpd.GeoDataFrame(geometry=pts, crs=edges.crs)
    gdf_pts = gdf_pts.to_crs(epsg=4326)
    out = pd.DataFrame({
        "LOC_LAT": gdf_pts.geometry.y,
        "LOC_LONG": gdf_pts.geometry.x,
    })
    return out

def sample_negative_coords(road_points_df: pd.DataFrame, n_neg: int, rng=42):
    """
    Parameters:
        - road_points_df: dataframe with road points collected
        - n_neg         : # of negative points we want
        - rng           : random seed

    Randomly sample N coordinates from the road points pool.
    Returns a DataFrame with LOC_LAT, LOC_LONG, is_accident=0.
    """
    rs = np.random.RandomState(rng)
    replace = len(road_points_df) < n_neg
    idx = rs.choice(len(road_points_df), size=n_neg, replace=replace)
    neg = road_points_df.iloc[idx].copy().reset_index(drop=True)
    neg["is_accident"] = 0
    return neg

def month_to_season(m):
    """
    Parameters:
        - m: Month
    Create season var using month var.
    """

    # 1=winter, 2=spring, 3=summer, 4=fall
    return int(((m % 12) // 3) + 1)

def build_season_pools(df_pos, cols):
    """
        Parameters:
        - df_pos: dataframe with positive cases (accidents)
        - cols  : list of columns 

    For each column in `cols`, build value pools per season from positives.
    Returns: {col: {season: np.array(values)}, 'overall': {col: np.array(values)}}.
    """
    pools = {'overall': {}}
    for col in cols:
        pools[col] = {}
        for s in (1, 2, 3, 4):
            vals = df_pos.loc[df_pos['season'] == s, col].dropna().values
            if len(vals) > 0:
                pools[col][s] = vals
        pools['overall'][col] = df_pos[col].dropna().values
    return pools

def sample_from_pool(pools, col, season, size, rng):
    """
    Parameters:
        - pools  : dict of pools built by build_season_pools
                    Structure: {col: {season: np.array(values)}, 'overall': {col: np.array(values)}}.
        - col    : column name to sample values for
        - season : season identifier (1=winter, 2=spring, 3=summer, 4=fall)
        - size   : number of values to sample
        - rng    : np.random.RandomState for reproducibility

    Sample `size` values for `col` using the season pool if available, else overall."""
    if season in pools[col] and len(pools[col][season]) > 0:
        base = pools[col][season]
    else:
        base = pools['overall'][col]
    return rng.choice(base, size=size)


def negative_cases(df_pos, spacing_m=100, ratio=1.0, seed=42):
    """
        Parameters:
        - df_pos    : dataframe with positive cases (accidents)
        - spacing_m : Spacing between each point simulated on each street
        - ratio     : ratio between negative and positive cases
        - seed      : Random seed

    Create negatives:
        - coordinates sampled on roads,
        - time sampled from positives (then flags recomputed),
        - context columns sampled *conditional on season* from positives.
    """
    rng = np.random.RandomState(seed)
    n_neg = int(len(df_pos) * ratio)

    # Sampling Coordinates for negative cases
    road_points = get_road_points(spacing_m=spacing_m, location="Montréal, Quebec, Canada")
    replace = len(road_points) < n_neg
    idx = rng.choice(len(road_points), size=n_neg, replace=replace)
    df_neg = road_points.iloc[idx].reset_index(drop=True)
    df_neg['is_accident'] = 0

    # Sample time variables (sample from positives, then recompute flags)
    df_neg['year']   = rng.choice(df_pos['year'].values,   size=n_neg)
    df_neg['month']  = rng.choice(df_pos['month'].values,  size=n_neg)
    # choose any valid day safely (1–28 avoids month-length issues)
    df_neg['day']    = rng.randint(1, 29, size=n_neg)
    df_neg['hour']   = rng.choice(df_pos['hour'].values,   size=n_neg)
    df_neg['weekday']= rng.choice(df_pos['weekday'].values,size=n_neg)

    df_neg['season']     = df_neg['month'].apply(month_to_season).astype(int)
    df_neg['is_weekend'] = (df_neg['weekday'] >= 5).astype(int)
    df_neg['is_rush_hour']= (df_neg['hour'].between(7, 9) | df_neg['hour'].between(15, 18)).astype(int)
    df_neg['is_night']   = ((df_neg['hour'] < 6) | (df_neg['hour'] > 20)).astype(int)

    # Context variables sampled conditional on season
    context_cols = [
        'CD_CATEG_ROUTE', 'CD_SIT_PRTCE_ACCDN',
        'CD_COND_METEO', 'CD_ETAT_SURFC', 'CD_ECLRM', 'CD_ENVRN_ACCDN'
    ]
    # only keep the ones present in df_pos
    context_cols = [c for c in context_cols if c in df_pos.columns]

    pools = build_season_pools(df_pos, context_cols)

    # sample per season chunk to preserve consistency
    for s in (1, 2, 3, 4):
        mask = df_neg['season'] == s
        k = int(mask.sum())
        if k == 0:
            continue
        for col in context_cols:
            df_neg.loc[mask, col] = sample_from_pool(pools, col, season=s, size=k, rng=rng)

    keep_cols = [
        'LOC_LAT', 'LOC_LONG',
        'CD_CATEG_ROUTE', 'CD_SIT_PRTCE_ACCDN', 'CD_COND_METEO',
        'CD_ETAT_SURFC', 'CD_ECLRM', 'CD_ENVRN_ACCDN',
        'year', 'month', 'day', 'hour', 'weekday', 'is_weekend',
        'is_rush_hour', 'season', 'is_night',
        'is_accident'
    ]
    keep_cols = [c for c in keep_cols if c in df_neg.columns]
    return df_neg[keep_cols]



def main():
    # Load positives (accidents) with only variables that make sense for both classes
    df_pos = load_raw()
    keep_cols = [
        "DT_ACCDN", "JR_SEMN_ACCDN", "HEURE_ACCDN",          # time 
        "LOC_LAT", "LOC_LONG",                                # coords
        "CD_CATEG_ROUTE", "CD_SIT_PRTCE_ACCDN",               # road / site
        "CD_COND_METEO", "CD_ETAT_SURFC", "CD_ECLRM", "CD_ENVRN_ACCDN"  # context
    ]
    df_pos = df_pos[keep_cols].copy()
    df_pos["is_accident"] = 1

    #  create date/time variables + handle missing data for positive cases
    df_pos = handle_time(df_pos)
    df_pos = missing_data(df_pos)

    # Generate negative cases
    df_neg = negative_cases(df_pos, spacing_m=100, ratio=1.0, seed=42)  

    # Align columns and concatenate
    common_cols = sorted(set(df_pos.columns).intersection(set(df_neg.columns)))
    df_all = pd.concat([df_pos[common_cols], df_neg[common_cols]], ignore_index=True)

    # confirm shape of both dataframes
    print("\nAccidents:", df_pos.shape, "Negatives:", df_neg.shape, "Combined:", df_all.shape)

    # One‑hot encode after the merge 
    df_all_encoded = encode(df_all)

    # Save combined dataset for modeling 
    save_processed(df_all_encoded)

if __name__ == "__main__":
    main()