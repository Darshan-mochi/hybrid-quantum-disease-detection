
import pandas as pd

# Names of the 14 columns in the Cleveland dataset
columns = [
    "age", "sex", "cp", "trestbps", "chol",
    "fbs", "restecg", "thalach", "exang",
    "oldpeak", "slope", "ca", "thal", "target"
]

# Read the dataset
df = pd.read_csv(
    "data/processed.cleveland.data",
    header=None,
    names=columns,
    na_values="?"
)

# Show the first five records
print("First five records:")
print(df.head())

# Check the size
print("\nDataset shape:", df.shape)

# Show the column names
print("\nColumn names:")
print(df.columns.tolist())

# Count missing values
print("\nMissing values:")
print(df.isnull().sum())

# Show the original target distribution
print("\nTarget distribution:")
print(df["target"].value_counts().sort_index())
