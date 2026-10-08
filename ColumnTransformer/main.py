import pandas as pd
from sklearn.model_selection import train_test_split
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


df=pd.read_csv("loan_data.csv")

df["employment_type"] = (
    df["employment_type"]
    .str.strip()
    .str.lower()
)

X = df.drop(
    ["default", "customer_id", "application_date"],
    axis=1
)

y = df["default"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

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

numeric_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])
categorical_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("num", numeric_transformer, numeric_features),
    ("cat", categorical_transformer, categorical_features)
])

# preprocessor.fit(X_train)
# X_train_processed = preprocessor.transform(X_train)
# X_test_processed = preprocessor.transform(X_test)

# print("Original X_train:", X_train.shape)
# print("Original X_test:", X_test.shape)

# print("Processed X_train:", X_train_processed.shape)
# print("Processed X_test:", X_test_processed.shape)

# feature_names = preprocessor.get_feature_names_out()

# print("Total features:", len(feature_names))

# for feature in feature_names:
#     print(feature)

# print(df["employment_type"].value_counts())

model = LogisticRegression(max_iter=1000)
model_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", model)
])

model_pipeline.fit(X_train, y_train)
y_pred = model_pipeline.predict(X_test)

print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)

print("\nActual values:")
print(y_test.values)

print("\nPredicted values:")
print(y_pred)

print("Accuracy:", accuracy_score(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))