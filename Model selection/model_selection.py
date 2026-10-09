
# import pandas as pd

# from sklearn.model_selection import train_test_split
# from sklearn.compose import ColumnTransformer
# from sklearn.preprocessing import StandardScaler, OneHotEncoder
# from sklearn.pipeline import Pipeline
# from sklearn.metrics import (
#     accuracy_score,
#     precision_score,
#     recall_score,
#     f1_score
# )

# from sklearn.dummy import DummyClassifier
# from sklearn.linear_model import LogisticRegression
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
# from sklearn.svm import SVC


# # 1. Load dataset
# df = pd.read_csv("employee_attrition_model_selection.csv")

# # 2. Separate features and target
# X = df.drop(columns=["attrition"])
# y = df["attrition"]

# # 3. Identify column types
# numeric_features = X.select_dtypes(include=["int64", "float64"]).columns
# categorical_features = X.select_dtypes(include=["object", "category"]).columns

# # 4. Preprocess numerical and categorical columns
# preprocessor = ColumnTransformer(
#     transformers=[
#         ("num", StandardScaler(), numeric_features),
#         ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
#     ]
# )

# # 5. Use the same split for every model
# X_train, X_test, y_train, y_test = train_test_split(
#     X,
#     y,
#     test_size=0.2,
#     random_state=42,
#     stratify=y
# )

# # 6. Define models
# models = {
#     "Dummy Baseline": DummyClassifier(strategy="most_frequent"),
#     "Logistic Regression": LogisticRegression(max_iter=1000),
#     "Decision Tree": DecisionTreeClassifier(
#         max_depth=5, random_state=42
#     ),
#     "Random Forest": RandomForestClassifier(
#         n_estimators=200, random_state=42
#     ),
#     "SVM": SVC(),
#     "Gradient Boosting": GradientBoostingClassifier(random_state=42)
# }

# # 7. Train and evaluate each model
# results = []

# for name, model in models.items():
#     pipeline = Pipeline([
#         ("preprocessing", preprocessor),
#         ("model", model)
#     ])

#     pipeline.fit(X_train, y_train)
#     predictions = pipeline.predict(X_test)

#     results.append({
#         "Model": name,
#         "Accuracy": accuracy_score(y_test, predictions),
#         "Precision": precision_score(
#             y_test, predictions, zero_division=0
#         ),
#         "Recall": recall_score(
#             y_test, predictions, zero_division=0
#         ),
#         "F1 Score": f1_score(
#             y_test, predictions, zero_division=0
#         )
#     })

# # 8. Display comparison
# results_df = pd.DataFrame(results)
# results_df = results_df.sort_values(
#     by="Recall", ascending=False
# )

# print("\nModel Comparison (sorted by Recall):")
# print(results_df.round(3).to_string(index=False))











import pandas as pd

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_predict
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

# 1. Load data
df = pd.read_csv("employee_attrition_model_selection.csv")

X = df.drop(columns=["attrition"])
y = df["attrition"]

# 2. Identify column types
numeric_features = X.select_dtypes(include=["number"]).columns
categorical_features = X.select_dtypes(
    include=["object", "category", "str"]
).columns

# 3. Preprocessing
preprocessor = ColumnTransformer([
    ("num", StandardScaler(), numeric_features),
    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
])

# 4. Same stratified split for every experiment
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# 5. Define models
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),

    "Logistic Regression + Class Weights": LogisticRegression(
        max_iter=1000,
        class_weight="balanced"
    ),

    "Random Forest + Class Weights": RandomForestClassifier(
        n_estimators=200,
        class_weight="balanced",
        random_state=42
    )
}

# Use training data only to tune thresholds
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

results = []

for name, model in models.items():
    pipeline = Pipeline([
        ("preprocessing", preprocessor),
        ("model", model)
    ])

    # Out-of-fold probabilities on training data
    train_probabilities = cross_val_predict(
        pipeline,
        X_train,
        y_train,
        cv=cv,
        method="predict_proba"
    )[:, 1]

    # Choose the threshold that maximizes training CV F1
    thresholds = [i / 100 for i in range(10, 91, 5)]

    threshold_scores = []
    for threshold in thresholds:
        train_predictions = (
            train_probabilities >= threshold
        ).astype(int)

        score = f1_score(
            y_train,
            train_predictions,
            zero_division=0
        )
        threshold_scores.append(score)

    best_index = threshold_scores.index(max(threshold_scores))
    best_threshold = thresholds[best_index]

    # Train on all training data
    pipeline.fit(X_train, y_train)

    # Predict probabilities on the untouched test set
    test_probabilities = pipeline.predict_proba(X_test)[:, 1]

    # Evaluate default threshold and tuned threshold
    thresholds_to_test = {
        "Default threshold (0.50)": 0.50,
        "CV-tuned threshold": best_threshold
    }

    for threshold_name, threshold in thresholds_to_test.items():
        predictions = (test_probabilities >= threshold).astype(int)

        results.append({
            "Model": name,
            "Threshold type": threshold_name,
            "Threshold": threshold,
            "Accuracy": accuracy_score(y_test, predictions),
            "Precision": precision_score(
                y_test, predictions, zero_division=0
            ),
            "Recall": recall_score(
                y_test, predictions, zero_division=0
            ),
            "F1 Score": f1_score(
                y_test, predictions, zero_division=0
            )
        })

        print("\n" + "=" * 60)
        print(name, "|", threshold_name)
        print("Threshold:", round(threshold, 2))
        print("\nConfusion matrix:")
        print(confusion_matrix(y_test, predictions))
        print("\nClassification report:")
        print(classification_report(
            y_test,
            predictions,
            zero_division=0
        ))

# 6. Compare all experiments
results_df = pd.DataFrame(results)

print("\n\nFINAL MODEL COMPARISON")
print(
    results_df.sort_values(
        by="Recall", ascending=False
    ).round(3).to_string(index=False)
)
