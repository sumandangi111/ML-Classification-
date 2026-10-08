import pandas as pd

from sklearn.model_selection import train_test_split, cross_val_score

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)


# ============================================================
# 1. LOAD DATA
# ============================================================

df = pd.read_csv("loan_data.csv")


# ============================================================
# 2. CLEAN DATA
# ============================================================

df["employment_type"] = (
    df["employment_type"]
    .str.strip()
    .str.lower()
)


# ============================================================
# 3. X AND y
# ============================================================

X = df.drop(
    ["default", "customer_id", "application_date"],
    axis=1
)

y = df["default"]


# ============================================================
# 4. TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# ============================================================
# 5. FEATURE TYPES
# ============================================================

numeric_features = [
    "age",
    "income",
    "employment_years",
    "loan_amount",
    "interest_rate",
    "credit_score",
    "previous_defaults",
    "monthly_debt",
    "loan_term"
]

categorical_features = [
    "employment_type",
    "city"
]


# ============================================================
# 6. NUMERICAL PREPROCESSING
# ============================================================

numeric_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])


# ============================================================
# 7. CATEGORICAL PREPROCESSING
# ============================================================

categorical_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])


# ============================================================
# 8. COLUMN TRANSFORMER
# ============================================================

preprocessor = ColumnTransformer([
    ("num", numeric_transformer, numeric_features),
    ("cat", categorical_transformer, categorical_features)
])


# ============================================================
# 9. MODEL
# ============================================================

model = LogisticRegression(max_iter=1000)


# ============================================================
# 10. COMPLETE PIPELINE
# ============================================================

model_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", model)
])


# ============================================================
# 11. CROSS-VALIDATION
# ============================================================

cv_scores = cross_val_score(
    model_pipeline,
    X_train,
    y_train,
    cv=5,
    scoring="accuracy"
)

print("\nCross-Validation Scores:")
print(cv_scores)

print("\nMean CV Accuracy:")
print(cv_scores.mean())


# ============================================================
# 12. TRAIN FINAL MODEL
# ============================================================

model_pipeline.fit(
    X_train,
    y_train
)


# ============================================================
# 13. TEST SET PREDICTION
# ============================================================

y_pred = model_pipeline.predict(
    X_test
)


# ============================================================
# 14. FINAL TEST EVALUATION
# ============================================================

print("\nTest Accuracy:")
print(accuracy_score(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))










import pandas as pd

from sklearn.model_selection import (
    train_test_split,
    cross_val_score,
    StratifiedKFold
)

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)


# ============================================================
# 1. LOAD DATA
# ============================================================

df = pd.read_csv("loan_data.csv")


# ============================================================
# 2. DATA CLEANING
# ============================================================

df["employment_type"] = (
    df["employment_type"]
    .str.strip()
    .str.lower()
)


# ============================================================
# 3. X AND y
# ============================================================

X = df.drop(
    ["default", "customer_id", "application_date"],
    axis=1
)

y = df["default"]


# ============================================================
# 4. TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# ============================================================
# 5. DEFINE FEATURES
# ============================================================

numeric_features = [
    "age",
    "income",
    "employment_years",
    "loan_amount",
    "interest_rate",
    "credit_score",
    "previous_defaults",
    "monthly_debt",
    "loan_term"
]

categorical_features = [
    "employment_type",
    "city"
]


# ============================================================
# 6. NUMERICAL PREPROCESSING
# ============================================================

numeric_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])


# ============================================================
# 7. CATEGORICAL PREPROCESSING
# ============================================================

categorical_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])


# ============================================================
# 8. COLUMN TRANSFORMER
# ============================================================

preprocessor = ColumnTransformer([
    ("num", numeric_transformer, numeric_features),
    ("cat", categorical_transformer, categorical_features)
])


# ============================================================
# 9. MODEL
# ============================================================

model = LogisticRegression(
    max_iter=1000
)


# ============================================================
# 10. COMPLETE PIPELINE
# ============================================================

model_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", model)
])


# ============================================================
# 11. STRATIFIED CROSS-VALIDATION
# ============================================================

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)


# ============================================================
# 12. CROSS-VALIDATION
# ============================================================

cv_scores = cross_val_score(
    model_pipeline,
    X_train,
    y_train,
    cv=cv,
    scoring="accuracy"
)


print("\nStratified CV Scores:")
print(cv_scores)

print("\nMean CV Accuracy:")
print(cv_scores.mean())


# ============================================================
# 13. TRAIN FINAL MODEL
# ============================================================

model_pipeline.fit(
    X_train,
    y_train
)


# ============================================================
# 14. TEST SET PREDICTION
# ============================================================

y_pred = model_pipeline.predict(
    X_test
)


# ============================================================
# 15. FINAL TEST EVALUATION
# ============================================================

print("\nTest Accuracy:")
print(accuracy_score(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))