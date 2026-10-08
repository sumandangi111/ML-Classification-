# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns

# # Load
# df = pd.read_csv("loan_data.csv")

# # 1. Shape
# print("Shape:", df.shape)

# # 2. First rows
# print(df.head())

# # 3. Information
# df.info()

# # 4. Target distribution
# print(df["default"].value_counts())
# print(df["default"].value_counts(normalize=True) * 100)

# # 5. Missing values
# print(df.isnull().sum())

# # 6. Duplicates
# print("Duplicates:", df.duplicated().sum())

# # 7. Unique values
# print(df.nunique())

# # 8. Numerical summary
# print(df.describe())

# # 9. Categorical summary
# print(df["employment_type"].value_counts())

# # 10. Feature vs target
# print(
#     df.groupby("default")["credit_score"].mean()
# )

# # 11. Categorical vs target
# print(
#     df.groupby("employment_type")["default"].mean()
# )

# # 12. Correlation
# corr = df.select_dtypes("number").corr()
# print(corr["default"].sort_values())


# # Target
# sns.countplot(x="default", data=df)
# plt.show()

# # Income
# sns.histplot(df["income"], kde=True)
# plt.show()

# # Credit score vs default
# sns.boxplot(
#     x="default",
#     y="credit_score",
#     data=df
# )
# plt.show()

# # Loan amount vs default
# sns.boxplot(
#     x="default",
#     y="loan_amount",
#     data=df
# )
# plt.show()

# # Employment type
# sns.countplot(
#     x="employment_type",
#     data=df
# )
# plt.xticks(rotation=45)
# plt.show()

# # Correlation
# sns.heatmap(
#     df.select_dtypes("number").corr(),
#     annot=True
# )
# plt.show()



import pandas as pd
import numpy as np

df = pd.read_csv("loan_data.csv")

# 1. Remove exact duplicate rows
df = df.drop_duplicates()

# 2. Clean categorical text
df["employment_type"] = (
    df["employment_type"]
    .str.strip()
    .str.lower()
)

# 3. Convert date
df["application_date"] = pd.to_datetime(
    df["application_date"]
)

# 4. Handle impossible values
df.loc[df["income"] < 0, "income"] = np.nan

df.loc[
    (df["credit_score"] < 300) |
    (df["credit_score"] > 850),
    "credit_score"
] = np.nan

# 5. Check the result
print(df.isnull().sum())
print(df.duplicated().sum())
print(df.dtypes)