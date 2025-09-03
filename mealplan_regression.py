import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score

# --------------------
# 1. Load Data
# --------------------
path = "C:/Users/siddh/dataset/Meal_Plan.csv"  # adjust filename as per dataset

df = pd.read_csv(path)
print("Data Shape:", df.shape)
print(df.head())

# --------------------
# 2. Preprocessing
# --------------------
# Encode categorical variables
for col in df.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])

# Define features and target (example: predict 'Calories')
if 'BMI ' in df.columns:
    target = 'BMI '
elif 'Meal Plan' in df.columns:  # alternative target
    target = 'Meal Plan'
else:
    raise ValueError("Please check dataset and set a numeric target column (e.g., Calories, Duration)")

X = df.drop(target, axis=1)
y = df[target]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardization
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# --------------------
# 3. Models
# --------------------
results = {}

# (a) Linear Regression
lin_reg = LinearRegression()
lin_reg.fit(X_train_scaled, y_train)
y_pred = lin_reg.predict(X_test_scaled)
results['Linear Regression'] = (mean_squared_error(y_test, y_pred), r2_score(y_test, y_pred))

# (b) Polynomial Regression (degree=2)
poly = PolynomialFeatures(degree=2)
X_train_poly = poly.fit_transform(X_train_scaled)
X_test_poly = poly.transform(X_test_scaled)
poly_reg = LinearRegression()
poly_reg.fit(X_train_poly, y_train)
y_pred = poly_reg.predict(X_test_poly)
results['Polynomial Regression (deg=2)'] = (mean_squared_error(y_test, y_pred), r2_score(y_test, y_pred))

# (c) Decision Tree Regressor
tree_reg = DecisionTreeRegressor(random_state=42, max_depth=10)
tree_reg.fit(X_train_scaled, y_train)
y_pred = tree_reg.predict(X_test_scaled)
results['Decision Tree'] = (mean_squared_error(y_test, y_pred), r2_score(y_test, y_pred))

# (d) Random Forest Regressor
rf_reg = RandomForestRegressor(n_estimators=100, random_state=42)
rf_reg.fit(X_train_scaled, y_train)
y_pred = rf_reg.predict(X_test_scaled)
results['Random Forest'] = (mean_squared_error(y_test, y_pred), r2_score(y_test, y_pred))

# (e) Neural Network (MLP Regressor)
nn_reg = MLPRegressor(hidden_layer_sizes=(64,64), max_iter=500, random_state=42)
nn_reg.fit(X_train_scaled, y_train)
y_pred = nn_reg.predict(X_test_scaled)
results['Neural Network'] = (mean_squared_error(y_test, y_pred), r2_score(y_test, y_pred))

# --------------------
# 4. Results Comparison
# --------------------
results_df = pd.DataFrame(results, index=['MSE', 'R2']).T
print("\nModel Performance Comparison:")
print(results_df)

# Visualization
plt.figure(figsize=(8,5))
sns.barplot(x=results_df.index, y=results_df['R2'])
plt.title("Model Comparison (R2 Score)")
plt.ylabel("R2 Score")
plt.xticks(rotation=45)
plt.show()
