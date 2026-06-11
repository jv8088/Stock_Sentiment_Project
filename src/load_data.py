import pandas as pd

# Load dataset
df = pd.read_csv(
    "data/tweets.csv",
    encoding="latin-1",
    header=None
)

# Display first 5 rows
print(df.head())

# Display shape
print("\nShape:", df.shape)