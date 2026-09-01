"""
Step 2 — Data Cleaning
Handles missing values, duplicates, outliers, type fixes.
Saves cleaned CSV to data_cleaned.csv.
"""
import pandas as pd

from config import DATA_CLEANED, DATA_RAW, ID_COLUMN, NUMERIC_PLOT_COLUMNS, TARGET, TEXT_COLUMNS

df = pd.read_csv(DATA_RAW, na_values=["NULL", "null", "N/A", "NA", "nan", "NaN", "-"])
print("Raw shape:", df.shape)

# Convert expected numeric fields before checking and filling missing values.
expected_numeric_columns = NUMERIC_PLOT_COLUMNS + [ID_COLUMN, TARGET]
for column in expected_numeric_columns:
    df[column] = pd.to_numeric(df[column], errors="coerce")

# ── 1. Drop duplicates ──────────────────────────────────────────────────
before = len(df)
df.drop_duplicates(inplace=True)
print(f"Duplicates removed: {before - len(df)}")

# ── 2. Handle missing values ────────────────────────────────────────────
missing = df.isnull().sum()
print("\nMissing values per column:")
print(missing[missing > 0] if missing.any() else "None")

numeric_columns = expected_numeric_columns.copy()
cat_cols = TEXT_COLUMNS.copy()

for col in numeric_columns:
    if df[col].isnull().any():
        df[col] = df[col].fillna(df[col].median())

for col in cat_cols:
    if df[col].isnull().any():
        df[col] = df[col].fillna(df[col].mode()[0])

# ── 3. Outlier capping (1st/99th percentile method) ─────────────────────
cols_to_cap = NUMERIC_PLOT_COLUMNS

for col in cols_to_cap:
    q1 = df[col].quantile(0.01)
    q99 = df[col].quantile(0.99)
    before_clip = ((df[col] < q1) | (df[col] > q99)).sum()
    df[col] = df[col].clip(q1, q99)
    if before_clip > 0:
        print(f"Clipped {before_clip} outliers in '{col}'")

# ── 4. Encode categorical features ──────────────────────────────────────
df[TEXT_COLUMNS[0]] = df[TEXT_COLUMNS[0]].map({"Male": 0, "Female": 1})

df = pd.get_dummies(df, columns=TEXT_COLUMNS[1:],
                    drop_first=True, dtype=int)

# ── 5. Drop CustomerID (not a feature) ──────────────────────────────────
df.drop(columns=[ID_COLUMN], inplace=True)

# ── 6. Save ─────────────────────────────────────────────────────────────
df.to_csv(DATA_CLEANED, index=False)
print(f"\nCleaned shape: {df.shape}")
print(f"Saved to: {DATA_CLEANED}")
print(f"Remaining missing values: {df.isna().sum().sum()}")
print(f"Remaining duplicate rows: {df.duplicated().sum()}")
print(f"Remaining text columns: {df.select_dtypes(include='str').columns.tolist()}")
print("\nData cleaning complete.")