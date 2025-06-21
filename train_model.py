import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score
import joblib

# Load dataset
df = pd.read_csv(r"C:\Users\aryan\Downloads\archive (1)\Laptop_price.csv")

# ✅ Optional: Remove very cheap laptops (less than ₹10,000)
df = df[df["Price"] >= 10000]

# Features and target
X = df.drop("Price", axis=1)

# ✅ Log transform the target
y = np.log1p(df["Price"])

# Preprocessing
cat_features = ["Brand"]
num_features = ["Processor_Speed", "RAM_Size", "Storage_Capacity", "Screen_Size", "Weight"]

preprocessor = ColumnTransformer([
    ('cat', OneHotEncoder(handle_unknown='ignore'), cat_features),
    ('num', StandardScaler(), num_features)
])

# Use Gradient Boosting
model = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', GradientBoostingRegressor(n_estimators=200, learning_rate=0.1, random_state=42))
])

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit model
model.fit(X_train, y_train)

# Evaluate model (optional)
y_pred = model.predict(X_test)
print("R² Score:", r2_score(y_test, y_pred))

# Save the model
joblib.dump(model, "model.pkl")
