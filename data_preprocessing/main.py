# Load CSV
# df = pd.read_csv("employee_attrition.csv")

# 1. Create X
# 2. Create y
# 3. 80/20 train-test split
# 4. random_state=42
# 5. stratify=y
# 6. Print shapes

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
df=pd.read_csv("employee_attrition.csv")

X=df.drop("attrition",axis=1)
y=df["attrition"]

# print("X:")
# print(X.head())

# print("\ny:")
# print(y.head())

X_train, X_test, y_train, y_test=train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# print("X_train:", X_train.shape)
# print("X_test:", X_test.shape)

# print("y_train:")
# print(y_train.value_counts())

# print("y_test:")
# print(y_test.value_counts())

print(df.isnull().sum())