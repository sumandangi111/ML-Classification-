# import pandas as pd
# import numpy as np
# df=pd.read_csv("feature engineering\loan_data.csv")
# df["loan_to_income"]=(
#     df["loan_amount"]/df["income"]
# )
# df["debt_to_income"]=(
#     df["monthly_debt"]/df["income"]
# )
# df["loan_per_employment_year"]=(
#     df["loan_amount"]/
#     df["employment_years"].replace(0,1)
# )



# import pandas as pd
# import numpy as np
# df=pd.read_csv("ecommerce_data.csv")
# df["avg_time_per_visit"]=(
#     df["total_time_minutes"]/df["website_visits"]
# )
# df["avg_spending_per_order"]=(
#     df["total_spending"]/df["previous_orders"].replace(0, np.nan)
# )
# df["has_previous_order"]=(
#     df["previous_orders"]>0
# ).astype(int)

# print(df.head())
# print(df[[
#     "avg_time_per_visit",
#     "avg_spending_per_order",
#     "has_previous_order"
# ]].describe())



import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

df=pd.read_csv("loan_data.csv")
X = df.drop("default", axis=1)
y = df["default"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Training shape:", X_train.shape)
print("Testing shape:", X_test.shape)

print("Training target:")
print(y_train.value_counts(normalize=True))

print("Testing target:")
print(y_test.value_counts(normalize=True))