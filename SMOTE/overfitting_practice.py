
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score

# Load dataset
df = pd.read_csv("employee_attrition_dataset.csv")

X = df.drop(columns=["attrition"])
y = df["attrition"]

# Split data before training
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# Identify categorical and numerical columns
categorical_cols = X.select_dtypes(
    include=["object", "category", "str"]
).columns.tolist()

numerical_cols = X.select_dtypes(
    exclude=["object", "category", "str"]
).columns.tolist()

# Preprocessing
preprocessor = ColumnTransformer([
    ("num", "passthrough", numerical_cols),
    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols)
])

# Compare a simple tree with a complex tree
models = {
    "Shallow Tree": DecisionTreeClassifier(
        max_depth=2, random_state=42
    ),
    "Deep Tree": DecisionTreeClassifier(
        max_depth=None, random_state=42
    )
}

for name, model in models.items():
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", model)
    ])

    pipeline.fit(X_train, y_train)

    train_predictions = pipeline.predict(X_train)
    test_predictions = pipeline.predict(X_test)

    print(f"\n{name}")
    print("-" * 30)

    print(
        "Training Accuracy:",
        round(accuracy_score(y_train, train_predictions), 3)
    )
    print(
        "Testing Accuracy:",
        round(accuracy_score(y_test, test_predictions), 3)
    )
    print(
        "Training F1:",
        round(f1_score(y_train, train_predictions, zero_division=0), 3)
    )
    print(
        "Testing F1:",
        round(f1_score(y_test, test_predictions, zero_division=0), 3)
    )
