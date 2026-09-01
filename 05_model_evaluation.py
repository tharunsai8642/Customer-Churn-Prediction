"""
Step 5 — Model Evaluation
Loads the saved Decision Tree and Random Forest models,
compares their performance, and saves evaluation plots/files.
"""

import os
import pickle

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import cross_val_score, train_test_split

from config import DATA_CLEANED, OUTPUT_DIR, RANDOM_STATE, TARGET, TEST_SIZE

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load cleaned dataset

df = pd.read_csv(DATA_CLEANED)
X = df.drop(columns=[TARGET])
y = df[TARGET]

# Use the same train/test split as the trained models
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y,
)

# Load saved models
model_paths = {
    "Decision Tree": os.path.join(OUTPUT_DIR, "decision_tree_model.pkl"),
    "Random Forest": os.path.join(OUTPUT_DIR, "random_forest_model.pkl"),
}

models = {}
for name, path in model_paths.items():
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file not found: {path}. Train the model first.")
    with open(path, "rb") as model_file:
        models[name] = pickle.load(model_file)

# Build comparison table
rows = []
for name, model in models.items():
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    cv_scores = cross_val_score(model, X, y, cv=5, scoring="f1")

    rows.append(
        {
            "Model": name,
            "Accuracy": round(accuracy_score(y_test, y_pred), 4),
            "Precision": round(precision_score(y_test, y_pred), 4),
            "Recall": round(recall_score(y_test, y_pred), 4),
            "F1 Score": round(f1_score(y_test, y_pred), 4),
            "ROC-AUC": round(roc_auc_score(y_test, y_prob), 4),
            "CV F1 (mean)": round(cv_scores.mean(), 4),
            "CV F1 (std)": round(cv_scores.std(), 4),
        }
    )

comparison = pd.DataFrame(rows)
print("\n===== MODEL COMPARISON =====")
print(comparison.to_string(index=False))
comparison_path = os.path.join(OUTPUT_DIR, "model_comparison.csv")
comparison.to_csv(comparison_path, index=False)
print(f"Saved model comparison CSV to: {comparison_path}")

# ROC Curve comparison
plt.figure(figsize=(8, 6))
for name, model in models.items():
    y_prob = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    auc_score = roc_auc_score(y_test, y_prob)
    plt.plot(fpr, tpr, label=f"{name} (AUC={auc_score:.3f})")

plt.plot([0, 1], [0, 1], "k--", label="Random")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve Comparison")
plt.legend()
plt.tight_layout()
roc_path = os.path.join(OUTPUT_DIR, "roc_curve_comparison.png")
plt.savefig(roc_path, dpi=150)
plt.close()
print(f"Saved ROC curve comparison plot to: {roc_path}")

# Confusion Matrices side by side
fig, axes = plt.subplots(1, len(models), figsize=(12, 5))
if len(models) == 1:
    axes = [axes]

for ax, (name, model) in zip(axes, models.items()):
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    ConfusionMatrixDisplay(
        cm,
        display_labels=["Not Churned", "Churned"],
    ).plot(ax=ax)
    ax.set_title(name)

plt.tight_layout()
confusion_path = os.path.join(OUTPUT_DIR, "confusion_matrices.png")
plt.savefig(confusion_path, dpi=150)
plt.close()
print(f"Saved confusion matrix plot to: {confusion_path}")

# Optional classification report summary for each model
print("\n===== CLASSIFICATION REPORTS =====")
for name, model in models.items():
    y_pred = model.predict(X_test)
    print(f"\n{name}:\n{classification_report(y_test, y_pred, target_names=['Not Churned', 'Churned'])}")
