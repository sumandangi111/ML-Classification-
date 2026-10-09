
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

df = pd.read_csv("employee_attrition_model_selection.csv")

X = df.drop(columns=["attrition"])
y = df["attrition"]

numeric_features = X.select_dtypes(include=["number"]).columns
categorical_features = X.select_dtypes(
    include=["object", "category", "str"]
).columns

preprocessor = ColumnTransformer([
    ("num", StandardScaler(), numeric_features),
    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

models = {
    "Without class weights": LogisticRegression(max_iter=1000),
    "With balanced class weights": LogisticRegression(
        max_iter=1000,
        class_weight="balanced"
    )
}

for name, classifier in models.items():
    model = Pipeline([
        ("preprocessing", preprocessor),
        ("classifier", classifier)
    ])

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    print(f"\n--- {name} ---")
    print("Confusion matrix:")
    print(confusion_matrix(y_test, predictions))
    print("Accuracy:", round(accuracy_score(y_test, predictions), 3))
    print("Precision:", round(
        precision_score(y_test, predictions, zero_division=0), 3
    ))
    print("Recall:", round(
        recall_score(y_test, predictions, zero_division=0), 3
    ))
    print("F1:", round(
        f1_score(y_test, predictions, zero_division=0), 3
    ))
