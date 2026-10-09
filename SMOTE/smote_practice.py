
import pandas as pd

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

# 1. Load the dataset
df = pd.read_csv("employee_attrition_dataset.csv")

X = df.drop(columns=["attrition"])
y = df["attrition"]

print("Original class distribution:")
print(y.value_counts())

# 2. Identify numeric and categorical features
numeric_features = X.select_dtypes(include=["number"]).columns
categorical_features = X.select_dtypes(
    include=["object", "category", "str"]
).columns

# 3. Preprocess the features
preprocessor = ColumnTransformer([
    ("num", StandardScaler(), numeric_features),
    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
])

# 4. Split BEFORE applying SMOTE
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTraining distribution before SMOTE:")
print(y_train.value_counts())

print("\nTest distribution (unchanged):")
print(y_test.value_counts())

# 5. Build a pipeline:
# preprocess -> SMOTE -> logistic regression
model = ImbPipeline([
    ("preprocessing", preprocessor),
    ("smote", SMOTE(random_state=42)),
    ("classifier", LogisticRegression(max_iter=1000))
])

# 6. Train the model
model.fit(X_train, y_train)

# 7. Predict on the original, untouched test data
y_pred = model.predict(X_test)

# 8. Evaluate
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nAccuracy:", round(accuracy_score(y_test, y_pred), 3))
print("Precision:", round(
    precision_score(y_test, y_pred, zero_division=0), 3
))
print("Recall:", round(
    recall_score(y_test, y_pred, zero_division=0), 3
))
print("F1 Score:", round(
    f1_score(y_test, y_pred, zero_division=0), 3
))
