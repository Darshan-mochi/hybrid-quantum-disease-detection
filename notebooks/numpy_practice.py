
import numpy as np

# Cholesterol values from five example records
cholesterol = np.array([200, 240, 180, 260, 220])

# Display all values
print("Cholesterol:", cholesterol)

# Calculate average
print("Average:", np.mean(cholesterol))

# Find highest and lowest values
print("Maximum:", np.max(cholesterol))
print("Minimum:", np.min(cholesterol))

# Find array size
print("Total records:", len(cholesterol))
