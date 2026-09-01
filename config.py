"""Central configuration for paths and constants."""

import os


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_RAW = os.path.join(
    BASE_DIR,
    "customer_churn_dataset-testing-master.csv",
)
DATA_CLEANED = os.path.join(BASE_DIR, "data_cleaned.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

RAW_DATA_PATH = DATA_RAW
TARGET = "Churn"
TARGET_COLUMN = TARGET
ID_COLUMN = "CustomerID"

RANDOM_STATE = 42
TEST_SIZE = 0.2

TEXT_COLUMNS = [
    "Gender",
    "Subscription Type",
    "Contract Length",
]

NUMERIC_PLOT_COLUMNS = [
    "Age",
    "Tenure",
    "Usage Frequency",
    "Support Calls",
    "Payment Delay",
    "Total Spend",
    "Last Interaction",
]

CATEGORICAL_PLOT_COLUMNS = [
    "Gender",
    "Subscription Type",
    "Contract Length",
    TARGET,
]
