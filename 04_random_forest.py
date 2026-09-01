"""
Step 4 — Random Forest Classifier
Trains a Random Forest on the cleaned data and saves the model.
"""

import os
import pickle

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

from config import (
    DATA_CLEANED,
    OUTPUT_DIR,
    RANDOM_STATE,
    TARGET,
    TRAIN_SIZE,
    VALIDATION_SIZE,
    TEST_SIZE,
)

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load cleaned data
df = pd.read_csv(DATA_CLEANED)
X = df.drop(columns=[TARGET])
y = df[TARGET]

# 60% training, 20% validation, 20% testing
X_train, X_temp, y_train, y_temp = train_test_split(
    X,
    y,
    train_size=TRAIN_SIZE,
    random_state=RANDOM_STATE,
    stratify=y,
)

validation_test_ratio = VALIDATION_SIZE / (VALIDATION_SIZE + TEST_SIZE)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp,
    y_temp,
    test_size=1 - validation_test_ratio,
    random_state=RANDOM_STATE,
    stratify=y_temp,
)

print(f"Train: {X_train.shape}")
print(f"Validation: {X_val.shape}")
print(f"Test: {X_test.shape}")

# Create and train the Random Forest
rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=12,
    min_samples_split=10,
    random_state=RANDOM_STATE,
    n_jobs=-1,
)
rf.fit(X_train, y_train)

# Validate model
y_val_pred = rf.predict(X_val)
y_val_prob = rf.predict_proba(X_val)[:, 1]

print("\n===== VALIDATION RESULTS =====")
print(f"Accuracy : {accuracy_score(y_val, y_val_pred):.4f}")
print(f"Precision : {precision_score(y_val, y_val_pred):.4f}")
print(f"Recall : {recall_score(y_val, y_val_pred):.4f}")
print(f"F1 Score : {f1_score(y_val, y_val_pred):.4f}")
print(f"ROC-AUC : {roc_auc_score(y_val, y_val_prob):.4f}")

# Evaluate on final test data
y_pred = rf.predict(X_test)
y_prob = rf.predict_proba(X_test)[:, 1]

print("\n===== TEST RESULTS =====")
print(f"Accuracy : {accuracy_score(y_test, y_pred):.4f}")
print(f"Precision : {precision_score(y_test, y_pred):.4f}")
print(f"Recall : {recall_score(y_test, y_pred):.4f}")
print(f"F1 Score : {f1_score(y_test, y_pred):.4f}")
print(f"ROC-AUC : {roc_auc_score(y_test, y_prob):.4f}")
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=["Not Churned", "Churned"]))

# Feature importance chart
feat_imp = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=True)
plt.figure(figsize=(10, 6))
feat_imp.plot.barh()
plt.title("Random Forest - Feature Importance")
plt.tight_layout()
feature_importance_path = os.path.join(OUTPUT_DIR, "rf_feature_importance.png")
plt.savefig(feature_importance_path, dpi=150)
plt.close()
print(f"Saved feature importance plot to: {feature_importance_path}")

# Save model to disk
model_path = os.path.join(OUTPUT_DIR, "random_forest_model.pkl")
with open(model_path, "wb") as model_file:
    pickle.dump(rf, model_file)
print(f"Saved model to: {model_path}")
