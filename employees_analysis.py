# =============================
# HR Attrition Analysis Project
# =============================

# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# import warnings
# warnings.filterwarnings('ignore')
import warnings
warnings.filterwarnings("ignore")


# Set pandas display options
pd.set_option('display.max_columns', 35)

# Load dataset
df = pd.read_csv("WA_Fn-UseC_-HR-Employee-Attrition.csv")

# Quick look
print("Shape of dataset:", df.shape)
print(df.head())

# -----------------------------
# Data Cleaning
# -----------------------------

# Check duplicates
print("Duplicated rows:", df.duplicated().sum())

# Check missing values
print("Missing values (%):")
print(df.isnull().sum() / len(df) * 100)

# Data types
print("\nData types:")
print(df.dtypes)

# -----------------------------
# Exploratory Data Analysis (EDA)
# -----------------------------

# 1. Attrition distribution
plt.figure(figsize=(6,4))
sns.countplot(x="Attrition", data=df, palette="Set2")
plt.title("Employee Attrition Distribution")
plt.show()

# 2. Attrition by Department
plt.figure(figsize=(8,5))
sns.countplot(x="Department", hue="Attrition", data=df, palette="husl")
plt.title("Attrition by Department")
plt.xticks(rotation=45)
plt.show()

# 3. Attrition by Gender
plt.figure(figsize=(6,4))
sns.countplot(x="Gender", hue="Attrition", data=df, palette="coolwarm")
plt.title("Attrition by Gender")
plt.show()

# 4. Attrition by Age (Histogram)
plt.figure(figsize=(8,5))
sns.histplot(df[df["Attrition"]=="Yes"]["Age"], bins=20, kde=True, color="red", label="Attrition=Yes")
sns.histplot(df[df["Attrition"]=="No"]["Age"], bins=20, kde=True, color="green", label="Attrition=No")
plt.legend()
plt.title("Attrition by Age")
plt.show()

# 5. Monthly Income vs Attrition
plt.figure(figsize=(8,5))
sns.boxplot(x="Attrition", y="MonthlyIncome", data=df, palette="pastel")
plt.title("Monthly Income vs Attrition")
plt.show()

# 6. Work-Life Balance vs Attrition
plt.figure(figsize=(8,5))
sns.countplot(x="WorkLifeBalance", hue="Attrition", data=df, palette="viridis")
plt.title("Work-Life Balance vs Attrition")
plt.show()
