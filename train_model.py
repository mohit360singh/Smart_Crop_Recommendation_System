import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import pickle

# Load dataset
file_path = "data/Crop_recommendation.csv"
df = pd.read_csv(file_path)

print("SMART CROP RECOMMENDATION SYSTEM")
print("-" * 40)

# Separate input features and target
X = df.drop("label", axis=1)
y = df["label"]

# Split dataset into training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("\nTraining the Random Forest model...")

# Create and train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predict test data
y_pred = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)

print("\nModel trained successfully!")
print(f"Model Accuracy: {accuracy * 100:.2f}%")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Save trained model
with open("crop_model.pkl", "wb") as file:
    pickle.dump(model, file)

print("\nModel saved successfully as crop_model.pkl")