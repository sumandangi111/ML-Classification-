
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
from sklearn.dummy import DummyClassifier

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    recall_score,
    precision_score,
    f1_score
)


# ==========================================
# 1. LOAD DATASET
# ==========================================

df = pd.read_csv("loan_data.csv")

print("Dataset Shape:", df.shape)
print("\nFirst 5 Rows:")
print(df.head())


# ==========================================
# 2. DATA CLEANING
# ==========================================

df["employment_type"] = (
    df["employment_type"]
    .str.strip()
    .str.lower()
)

print("\nMissing Values:")
print(df.isnull().sum())


# ==========================================
# 3. SEPARATE FEATURES AND TARGET
# ==========================================

X = df.drop(
    ["default", "customer_id", "application_date"],
    axis=1
)

y = df["default"]

print("\nFeature Shape:", X.shape)
print("Target Distribution:")
print(y.value_counts())


# ==========================================
# 4. TRAIN/TEST SPLIT
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTraining Samples:", len(X_train))
print("Testing Samples:", len(X_test))


# ==========================================
# 5. PREPROCESSING
# ==========================================

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

# Numerical preprocessing
numeric_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

# Categorical preprocessing
categorical_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

# Combine both transformers
preprocessor = ColumnTransformer([
    ("num", numeric_transformer, numeric_features),
    ("cat", categorical_transformer, categorical_features)
])


# ==========================================
# 6. CREATE BASELINE MODEL
# ==========================================

dummy_model = Pipeline([
    ("preprocessor", preprocessor),
    ("model", DummyClassifier(strategy="most_frequent"))
])


# ==========================================
# 7. TRAIN DUMMY CLASSIFIER
# ==========================================

dummy_model.fit(X_train, y_train)

dummy_pred = dummy_model.predict(X_test)


# ==========================================
# 8. EVALUATE DUMMY CLASSIFIER
# ==========================================

print("\n" + "=" * 45)
print("DUMMY CLASSIFIER RESULTS")
print("=" * 45)

print(
    "Accuracy:",
    round(accuracy_score(y_test, dummy_pred), 4)
)

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, dummy_pred))

print("\nClassification Report:")
print(classification_report(
    y_test,
    dummy_pred,
    zero_division=0
))

print(
    "Default Precision:",
    round(precision_score(y_test, dummy_pred, zero_division=0), 4)
)

print(
    "Default Recall:",
    round(recall_score(y_test, dummy_pred, zero_division=0), 4)
)

print(
    "Default F1-score:",
    round(f1_score(y_test, dummy_pred, zero_division=0), 4)
)


# ==========================================
# 9. CREATE LOGISTIC REGRESSION MODEL
# ==========================================

logistic_model = Pipeline([
    ("preprocessor", preprocessor),
    ("model", LogisticRegression(max_iter=1000))
])


# ==========================================
# 10. TRAIN LOGISTIC REGRESSION
# ==========================================

logistic_model.fit(X_train, y_train)

logistic_pred = logistic_model.predict(X_test)


# ==========================================
# 11. EVALUATE LOGISTIC REGRESSION
# ==========================================

print("\n" + "=" * 45)
print("LOGISTIC REGRESSION RESULTS")
print("=" * 45)

print(
    "Accuracy:",
    round(accuracy_score(y_test, logistic_pred), 4)
)

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, logistic_pred))

print("\nClassification Report:")
print(classification_report(
    y_test,
    logistic_pred,
    zero_division=0
))

print(
    "Default Precision:",
    round(precision_score(y_test, logistic_pred, zero_division=0), 4)
)

print(
    "Default Recall:",
    round(recall_score(y_test, logistic_pred, zero_division=0), 4)
)

print(
    "Default F1-score:",
    round(f1_score(y_test, logistic_pred, zero_division=0), 4)
)


# ==========================================
# 12. STRATIFIED CROSS-VALIDATION
# ==========================================

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

dummy_cv_scores = cross_val_score(
    dummy_model,
    X_train,
    y_train,
    cv=cv,
    scoring="accuracy"
)

logistic_cv_scores = cross_val_score(
    logistic_model,
    X_train,
    y_train,
    cv=cv,
    scoring="accuracy"
)

print("\n" + "=" * 45)
print("CROSS-VALIDATION COMPARISON")
print("=" * 45)

print(
    "Dummy CV Accuracy:",
    round(dummy_cv_scores.mean(), 4)
)

print(
    "Logistic Regression CV Accuracy:",
    round(logistic_cv_scores.mean(), 4)
)


# ==========================================
# 13. FINAL MODEL COMPARISON
# ==========================================

comparison = pd.DataFrame({
    "Model": [
        "Dummy Classifier",
        "Logistic Regression"
    ],
    "CV Accuracy": [
        dummy_cv_scores.mean(),
        logistic_cv_scores.mean()
    ],
    "Test Accuracy": [
        accuracy_score(y_test, dummy_pred),
        accuracy_score(y_test, logistic_pred)
    ],
    "Default Precision": [
        precision_score(y_test, dummy_pred, zero_division=0),
        precision_score(y_test, logistic_pred, zero_division=0)
    ],
    "Default Recall": [
        recall_score(y_test, dummy_pred, zero_division=0),
        recall_score(y_test, logistic_pred, zero_division=0)
    ],
    "Default F1-score": [
        f1_score(y_test, dummy_pred, zero_division=0),
        f1_score(y_test, logistic_pred, zero_division=0)
    ]
})

print("\n" + "=" * 65)
print("FINAL MODEL COMPARISON")
print("=" * 65)

print(comparison.round(4).to_string(index=False))
