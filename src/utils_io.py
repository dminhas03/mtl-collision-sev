from pathlib import Path
import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[1] / "data"

# ========================
# RAW DATA (EDA STEP)
# ========================

# Load the original raw dataset
def load_raw():
    return pd.read_csv(DATA_DIR / "collisions_routieres.csv")


# ========================================
# FILTERED DATA (POST-EDA, PRE-PROCESSING)
# ========================================

# Save the filtered dataset after EDA/column selection
def save_filtered(df):
    (DATA_DIR / "collisions_filtered.parquet").parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(DATA_DIR / "collisions_filtered.parquet", index=False)

# Load the filtered dataset for preprocessing
def load_filtered():
    return pd.read_parquet(DATA_DIR / "collisions_filtered.parquet")


# ============================
# PROCESSED DATA (MODEL-READY)
# ===========================

# Save the fully processed dataset (after cleaning, encoding, feature engineering)
def save_processed(df):
    df.to_parquet(DATA_DIR / "collisions_processed.parquet", index=False)

# Load the processed dataset for modeling
def load_processed():
    return pd.read_parquet(DATA_DIR / "collisions_processed.parquet")