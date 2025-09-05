# -------------------------------------------------------------
# Customer Satisfaction Prediction Project
# Dataset: customer_support_tickets.csv
# Author: Krish Patel
# -------------------------------------------------------------

# 🎯 Objectives:
# 1. Clean and preprocess the dataset.
# 2. Perform Exploratory Data Analysis (EDA).
# 3. Build a Machine Learning model (Random Forest Classifier).
# 4. Evaluate model performance (Accuracy, Classification Report, Confusion Matrix).
# 5. Visualize top feature importances.

# -------------------------------------------------------------
# Step 1: Import Required Libraries
# -------------------------------------------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import warnings
warnings.filterwarnings("ignore")

# -------------------------------------------------------------
# Step 2: Load Dataset
# -------------------------------------------------------------
df = pd.read_csv("customer_support_tickets.csv")

print("✅ Dataset Loaded Successfully!")
print("Shape of dataset:", df.shape)
print("\nFirst 5 rows:")
print(df.head())

# -------------------------------------------------------------
# Step 3: Data Preprocessing
# -------------------------------------------------------------
# Check for missing values
print("\nMissing Values (%):")
print(df.isnull().sum() / len(df) * 100)

# Drop rows with missing values (if any)
df.dropna(inplace=True)

# Encode categorical variables
label_encoders = {}
for column in df.select_dtypes(include=["object"]).columns:
    if column not in ["CustomerID"]:  # Exclude CustomerID
        label_encoders[column] = LabelEncoder()
        df[column] = label_encoders[column].fit_transform(df[column])

print("\nData after preprocessing:")
print(df.head())

# -------------------------------------------------------------
# Step 4: Feature Selection
# -------------------------------------------------------------
X = df.drop(['Ticket ID', 'Customer Satisfaction Rating'], axis=1)
y = df["Customer Satisfaction Rating"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Feature Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# -------------------------------------------------------------
# Step 5: Model Building
# -------------------------------------------------------------
rfc = RandomForestClassifier(random_state=42)
rfc.fit(X_train, y_train)

# Predictions
y_pred = rfc.predict(X_test)

# -------------------------------------------------------------
# Step 6: Model Evaluation
# -------------------------------------------------------------
print("\n🔎 Model Evaluation:")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

# -------------------------------------------------------------
# Step 7: Visualization
# -------------------------------------------------------------
# Feature Importance
feature_importances = pd.Series(rfc.feature_importances_, index=X.columns)

plt.figure(figsize=(10, 6))
feature_importances.nlargest(10).plot(kind="barh", color="skyblue")
plt.title("Top 10 Feature Importances for Customer Satisfaction Prediction")
plt.xlabel("Importance Score")
plt.ylabel("Features")
plt.show()
