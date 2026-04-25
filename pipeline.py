import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import joblib
import os

# Load data
df = pd.read_csv("data/sensor_data.csv")

X = df.drop("Status", axis=1)
y = df["Status"]

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Scale
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)

# Train model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Save model
os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/sensor_model.pkl")
joblib.dump(scaler, "models/scaler.pkl")

print("✅ Model Trained Successfully")