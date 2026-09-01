import os

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import (
	CATEGORICAL_PLOT_COLUMNS,
	ID_COLUMN,
	NUMERIC_PLOT_COLUMNS,
	OUTPUT_DIR,
	RAW_DATA_PATH,
	TARGET_COLUMN,
)

df = pd.read_csv(RAW_DATA_PATH)
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("DATASET PREVIEW")
print(df.head())

print("\nDATASET SHAPE")
print(f"Rows: {df.shape[0]}")
print(f"Columns: {df.shape[1]}")

print("\nCOLUMN NAMES")
print(df.columns.tolist())

print("\nDATA TYPES")
print(df.dtypes)

print("\nMISSING VALUES BY COLUMN")
print(df.isna().sum())

print("\nMISSING VALUE PERCENTAGE BY COLUMN")
missing_percentage = (df.isna().mean() * 100).round(2)
print(missing_percentage)

print("\nNULL VALUE CHECK")
print(f"Total null values: {df.isnull().sum().sum()}")

print("\nEMPTY STRING VALUES BY COLUMN")
empty_strings = df.select_dtypes(include="str").apply(
	lambda column: column.fillna("").str.strip().eq("").sum()
)
print(empty_strings)

print("\nCOMMON TEXT PLACEHOLDERS")
placeholders = {"NULL", "null", "N/A", "NA", "na", "NaN", "nan", "-"}
placeholder_counts = df.astype(str).apply(
	lambda column: column.isin(placeholders).sum()
)
print(placeholder_counts)

print("\nDUPLICATE ROWS")
print(f"Number of duplicate rows: {df.duplicated().sum()}")

print("\nDUPLICATE CUSTOMER IDs")
print(f"Number of duplicate {ID_COLUMN}s: {df[ID_COLUMN].duplicated().sum()}")

print("\nUNIQUE VALUES BY COLUMN")
print(df.nunique(dropna=False))

print("\nCATEGORICAL VALUE COUNTS")
for column in df.select_dtypes(include="str").columns:
	print(f"\n{column}:")
	print(df[column].value_counts(dropna=False))

print("\nNUMERIC SUMMARY")
print(df.describe().T)

print("\nNEGATIVE VALUES BY NUMERIC COLUMN")
numeric_columns = df.select_dtypes(include="number").columns
negative_values = (df[numeric_columns] < 0).sum()
print(negative_values)

print("\nEXPECTED CATEGORY CHECK")
expected_categories = {
	"Gender": {"Male", "Female"},
	"Subscription Type": {"Basic", "Standard", "Premium"},
	"Contract Length": {"Monthly", "Quarterly", "Annual"},
}
for column, expected_values in expected_categories.items():
	actual_values = set(df[column].dropna().unique())
	invalid_values = actual_values - expected_values
	print(f"{column}: invalid values = {invalid_values or 'None'}")

print("\nPERCENTILE OUTLIER CHECK")
outlier_columns = NUMERIC_PLOT_COLUMNS
outlier_summary = []
for column in outlier_columns:
	lower_limit = df[column].quantile(0.01)
	upper_limit = df[column].quantile(0.99)
	outlier_count = ((df[column] < lower_limit) | (df[column] > upper_limit)).sum()
	outlier_summary.append({
		"Column": column,
		"Min": df[column].min(),
		"Max": df[column].max(),
		"1st Percentile": round(lower_limit, 2),
		"99th Percentile": round(upper_limit, 2),
		"Outliers": int(outlier_count),
	})
print(pd.DataFrame(outlier_summary).to_string(index=False))

print("\nCHURN COUNTS")
print(df[TARGET_COLUMN].value_counts(dropna=False))

print("\nCHURN PERCENTAGE")
print((df[TARGET_COLUMN].value_counts(normalize=True, dropna=False) * 100).round(2))

print("\nSAVING DISTRIBUTION PLOTS")
numeric_cols = [
	column for column in df.select_dtypes(include="number").columns
	if column != ID_COLUMN
]
fig, axes = plt.subplots(3, 3, figsize=(15, 12))
for ax, column in zip(axes.flatten(), numeric_cols):
	ax.hist(df[column].dropna(), bins=20, edgecolor="black")
	ax.set_title(column)
for ax in axes.flatten()[len(numeric_cols):]:
	ax.set_visible(False)
plt.tight_layout()
numeric_plot_path = os.path.join(OUTPUT_DIR, "numeric_distributions.png")
plt.savefig(numeric_plot_path, dpi=150)
plt.close()
print(f"Saved: {numeric_plot_path}")

cat_cols = df.select_dtypes(include="str").columns.tolist()
fig, axes = plt.subplots(1, len(cat_cols), figsize=(5 * len(cat_cols), 4))
if len(cat_cols) == 1:
	axes = [axes]
for ax, column in zip(axes, cat_cols):
	df[column].value_counts(dropna=False).plot.bar(ax=ax)
	ax.set_title(column)
	ax.tick_params(axis="x", rotation=45)
plt.tight_layout()
categorical_plot_path = os.path.join(OUTPUT_DIR, "categorical_distributions.png")
plt.savefig(categorical_plot_path, dpi=150)
plt.close()
print(f"Saved: {categorical_plot_path}")

correlation = df[numeric_cols].corr()
plt.figure(figsize=(10, 8))
image = plt.imshow(correlation, cmap="coolwarm", vmin=-1, vmax=1)
plt.colorbar(image)
plt.xticks(range(len(numeric_cols)), numeric_cols, rotation=45, ha="right")
plt.yticks(range(len(numeric_cols)), numeric_cols)
for row in range(len(numeric_cols)):
	for column in range(len(numeric_cols)):
		plt.text(column, row, f"{correlation.iloc[row, column]:.2f}",
				ha="center", va="center")
plt.title("Correlation Matrix")
plt.tight_layout()
correlation_plot_path = os.path.join(OUTPUT_DIR, "correlation_matrix.png")
plt.savefig(correlation_plot_path, dpi=150)
plt.close()
print(f"Saved: {correlation_plot_path}")

print("EDA complete.")
