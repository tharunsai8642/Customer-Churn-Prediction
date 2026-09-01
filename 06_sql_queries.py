"""
Step 6 — 15 SQL Queries on the churn dataset
Uses SQLite in-memory so no external DB server is needed.
"""

import sqlite3

import pandas as pd

from config import DATA_RAW

# Load CSV into SQLite in-memory database

df = pd.read_csv(DATA_RAW)
conn = sqlite3.connect(":memory:")
df.to_sql("customers", conn, index=False, if_exists="replace")

QUERIES = {
    "Q01 - Total number of customers": {
        "description": "Counts all rows in the table.",
        "sql": """
            SELECT COUNT(*) AS total_customers
            FROM customers;
        """,
    },
    "Q02 - Churn vs Non-Churn count": {
        "description": "Groups by Churn and counts each category.",
        "sql": """
            SELECT "Churn", COUNT(*) AS customer_count
            FROM customers
            GROUP BY "Churn"
            ORDER BY "Churn";
        """,
    },
    "Q03 - Average age of churned vs non-churned": {
        "description": "Calculates average, minimum, and maximum age by churn status.",
        "sql": """
            SELECT "Churn",
                   ROUND(AVG("Age"), 2) AS avg_age,
                   MIN("Age") AS min_age,
                   MAX("Age") AS max_age
            FROM customers
            GROUP BY "Churn"
            ORDER BY "Churn";
        """,
    },
    "Q04 - Churn rate by Subscription Type": {
        "description": "Shows which subscription plan has the highest churn rate.",
        "sql": """
            SELECT "Subscription Type",
                   COUNT(*) AS total_customers,
                   SUM("Churn") AS churned_customers,
                   ROUND(AVG("Churn") * 100, 2) AS churn_rate_pct
            FROM customers
            GROUP BY "Subscription Type"
            ORDER BY churn_rate_pct DESC;
        """,
    },
    "Q05 - Churn rate by Contract Length": {
        "description": "Shows which contract type has the highest churn rate.",
        "sql": """
            SELECT "Contract Length",
                   COUNT(*) AS total_customers,
                   SUM("Churn") AS churned_customers,
                   ROUND(AVG("Churn") * 100, 2) AS churn_rate_pct
            FROM customers
            GROUP BY "Contract Length"
            ORDER BY churn_rate_pct DESC;
        """,
    },
    "Q06 - Average Total Spend by Churn status": {
        "description": "Checks whether churned customers spend less on average.",
        "sql": """
            SELECT "Churn",
                   ROUND(AVG("Total Spend"), 2) AS avg_spend,
                   ROUND(SUM("Total Spend"), 2) AS total_spend
            FROM customers
            GROUP BY "Churn"
            ORDER BY "Churn";
        """,
    },
    "Q07 - Top 10 highest spending customers": {
        "description": "Displays the biggest spenders in the dataset.",
        "sql": """
            SELECT "CustomerID", "Age", "Gender",
                   "Total Spend", "Churn"
            FROM customers
            ORDER BY "Total Spend" DESC
            LIMIT 10;
        """,
    },
    "Q08 - Average Support Calls by Churn": {
        "description": "Compares support call volume between churned and non-churned customers.",
        "sql": """
            SELECT "Churn",
                   ROUND(AVG("Support Calls"), 2) AS avg_support_calls
            FROM customers
            GROUP BY "Churn"
            ORDER BY "Churn";
        """,
    },
    "Q09 - High vs Low support calls - churn rate": {
        "description": "Shows the churn rate for customers with more than 5 support calls versus others.",
        "sql": """
            SELECT CASE
                     WHEN "Support Calls" > 5 THEN 'High'
                     ELSE 'Low'
                   END AS support_level,
                   COUNT(*) AS total_customers,
                   SUM("Churn") AS churned_customers,
                   ROUND(AVG("Churn") * 100, 2) AS churn_rate_pct
            FROM customers
            GROUP BY support_level
            ORDER BY support_level;
        """,
    },
    "Q10 - Usage Frequency buckets and churn rate": {
        "description": "Groups customers by service usage level and compares churn rates.",
        "sql": """
            SELECT CASE
                     WHEN "Usage Frequency" <= 10 THEN '0-10'
                     WHEN "Usage Frequency" <= 20 THEN '11-20'
                     ELSE '21+'
                   END AS usage_bucket,
                   COUNT(*) AS total_customers,
                   ROUND(AVG("Churn") * 100, 2) AS churn_rate_pct
            FROM customers
            GROUP BY usage_bucket
            ORDER BY usage_bucket;
        """,
    },
    "Q11 - Average Tenure by Churn status": {
        "description": "Measures how long churned and non-churned customers stay with the service.",
        "sql": """
            SELECT "Churn",
                   ROUND(AVG("Tenure"), 2) AS avg_tenure
            FROM customers
            GROUP BY "Churn"
            ORDER BY "Churn";
        """,
    },
    "Q12 - Short vs Long tenure churn rate": {
        "description": "Compares churn rate between customers with less than 12 months and 12+ months tenure.",
        "sql": """
            SELECT CASE
                     WHEN "Tenure" < 12 THEN 'Short (<12m)'
                     ELSE 'Long (12m+)'
                   END AS tenure_group,
                   COUNT(*) AS total_customers,
                   SUM("Churn") AS churned_customers,
                   ROUND(AVG("Churn") * 100, 2) AS churn_rate_pct
            FROM customers
            GROUP BY tenure_group;
        """,
    },
    "Q13 - Churn rate by Gender": {
        "description": "Compares churn rate between male and female customers.",
        "sql": """
            SELECT "Gender",
                   COUNT(*) AS total_customers,
                   SUM("Churn") AS churned_customers,
                   ROUND(AVG("Churn") * 100, 2) AS churn_rate_pct
            FROM customers
            GROUP BY "Gender"
            ORDER BY churn_rate_pct DESC;
        """,
    },
    "Q14 - Payment delay and churn rate": {
        "description": "Shows whether customers with more payment delay are more likely to churn.",
        "sql": """
            SELECT CASE
                     WHEN "Payment Delay" <= 10 THEN 'Low delay'
                     WHEN "Payment Delay" <= 20 THEN 'Medium delay'
                     ELSE 'High delay'
                   END AS payment_delay_group,
                   COUNT(*) AS total_customers,
                   ROUND(AVG("Churn") * 100, 2) AS churn_rate_pct
            FROM customers
            GROUP BY payment_delay_group
            ORDER BY payment_delay_group;
        """,
    },
    "Q15 - Last interaction and churn trend": {
        "description": "Explores whether customer engagement level is associated with churn.",
        "sql": """
            SELECT CASE
                     WHEN "Last Interaction" <= 10 THEN 'Recent'
                     WHEN "Last Interaction" <= 20 THEN 'Moderate'
                     ELSE 'Old'
                   END AS interaction_group,
                   COUNT(*) AS total_customers,
                   ROUND(AVG("Churn") * 100, 2) AS churn_rate_pct
            FROM customers
            GROUP BY interaction_group
            ORDER BY interaction_group;
        """,
    },
}

for title, details in QUERIES.items():
    print("=" * 80)
    print(title)
    print(details["description"])
    print("-" * 80)
    result = pd.read_sql_query(details["sql"], conn)
    print(result.to_string(index=False))
    print()

conn.close()
print("All 15 SQL queries executed successfully.")
