
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

# Load the dataset
df = pd.read_csv("employee_attrition_model_selection.csv")

X = df.drop(columns=["attrition"])
y = df["attrition"]

# Split while preserving class proportions
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Baseline: always predict the majority class
model = DummyClassifier(strategy="most_frequent")
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Test-set class distribution:")
print(y_test.value_counts())

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nAccuracy:", round(accuracy_score(y_test, y_pred), 3))
print("Precision:", round(precision_score(y_test, y_pred, zero_division=0), 3))
print("Recall:", round(recall_score(y_test, y_pred, zero_division=0), 3))
print("F1 Score:", round(f1_score(y_test, y_pred, zero_division=0), 3))

print("\nClassification Report:")
print(classification_report(y_test, y_pred, zero_division=0))
