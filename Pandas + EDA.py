import pandas as pd

# load dataset
df = pd.read_csv("Nike_Sales_Uncleaned.csv")

print("\nFirst rows:")
print(df.head())

print("\nShape:", df.shape)

print("\nColumn info:")
print(df.info())

print("\nMissing values:")
print(df.isnull().sum())


# -------------------------------
# Clean column names and spaces
# -------------------------------

df.columns = df.columns.str.strip()

# remove spaces inside text cells
for col in df.select_dtypes(include="object").columns:
    df[col] = df[col].str.strip()


# -------------------------------
# Convert numeric columns
# -------------------------------

df["Units_Sold"] = pd.to_numeric(df["Units_Sold"], errors="coerce")
df["MRP"] = pd.to_numeric(df["MRP"], errors="coerce")
df["Revenue"] = pd.to_numeric(df["Revenue"], errors="coerce")
df["Discount_Applied"] = pd.to_numeric(df["Discount_Applied"], errors="coerce")


# -------------------------------
# Handle missing values
# -------------------------------

df["Units_Sold"] = df["Units_Sold"].fillna(df["Units_Sold"].median())
df["MRP"] = df["MRP"].fillna(df["MRP"].median())
df["Discount_Applied"] = df["Discount_Applied"].fillna(0)

df["Size"] = df["Size"].fillna("Unknown")


# -------------------------------
# Convert date column
# -------------------------------

df["Order_Date"] = pd.to_datetime(df["Order_Date"], errors="coerce")

df = df.dropna(subset=["Order_Date"])


# -------------------------------
# Remove duplicates
# -------------------------------

df = df.drop_duplicates()


# -------------------------------
# Final check
# -------------------------------

print("\nFinal dataset info:")
print(df.info())

print("\nMissing values after cleaning:")
print(df.isnull().sum())


# -------------------------------
# Save cleaned dataset
# -------------------------------

df.to_csv("nike_sales_cleaned.csv", index=False)

print("\nClean dataset saved as nike_sales_cleaned.csv")




from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor


# load cleaned data
df = pd.read_csv("nike_sales_cleaned.csv")

print("\nDataset loaded")
print(df.shape)


# choose features
X = df[[
    "Units_Sold",
    "MRP",
    "Discount_Applied",
    "Product_Line",
    "Sales_Channel",
    "Region",
    "Gender_Category"
]]

# target
y = df["Profit"]


# convert categorical columns
X = pd.get_dummies(X, drop_first=True)


# split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# ----------------------------
# Linear Regression
# ----------------------------

lr = LinearRegression()
lr.fit(X_train, y_train)

lr_pred = lr.predict(X_test)

print("\nLinear Regression results")
print("MAE:", mean_absolute_error(y_test, lr_pred))
print("R2:", r2_score(y_test, lr_pred))


# ----------------------------
# Decision Tree
# ----------------------------

dt = DecisionTreeRegressor()
dt.fit(X_train, y_train)

dt_pred = dt.predict(X_test)

print("\nDecision Tree results")
print("MAE:", mean_absolute_error(y_test, dt_pred))
print("R2:", r2_score(y_test, dt_pred))


# ----------------------------
# Random Forest
# ----------------------------

rf = RandomForestRegressor()
rf.fit(X_train, y_train)

rf_pred = rf.predict(X_test)

print("\nRandom Forest results")
print("MAE:", mean_absolute_error(y_test, rf_pred))
print("R2:", r2_score(y_test, rf_pred))


# quick look at predictions
results = pd.DataFrame({
    "Actual": y_test,
    "Predicted_RF": rf_pred
})

print("\nSample predictions:")
print(results.head())

import matplotlib.pyplot as plt

# get feature importance from the random forest model
importance = rf.feature_importances_

# create dataframe for easier reading
feature_importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": importance
})

# sort features
feature_importance = feature_importance.sort_values(
    by="Importance", ascending=False
)

print("\nFeature Importance:")
print(feature_importance)


# plot feature importance
plt.figure(figsize=(10,5))

plt.bar(feature_importance["Feature"],
        feature_importance["Importance"])

plt.xticks(rotation=90)

plt.title("Feature Importance (Random Forest)")
plt.xlabel("Features")
plt.ylabel("Importance Score")

plt.tight_layout()
plt.show()