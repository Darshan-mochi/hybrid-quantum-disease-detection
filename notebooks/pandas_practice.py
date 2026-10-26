
import pandas as pd

# Create a small example dataset
data = {
    "Age": [45, 60, 38, 55],
    "Cholesterol": [220, 280, 190, 260],
    "Disease": [0, 1, 0, 1]
}

# Convert dictionary to DataFrame
df = pd.DataFrame(data)

# Display the dataset
print(df)

# Display the first two rows
print("\nFirst two rows:")
print(df.head(2))

# Calculate average age
print("\nAverage age:", df["Age"].mean())

# Display dataset dimensions
print("\nShape:", df.shape)
