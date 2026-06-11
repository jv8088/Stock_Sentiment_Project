import pandas as pd

df = pd.read_csv(
    "data/tweets.csv",
    encoding="latin-1",
    header=None
)

df.columns = [
    "sentiment",
    "id",
    "date",
    "query",
    "user",
    "tweet"
]

print(df.head())
print("\nColumns:")
print(df.columns)
print("\nSentiment Counts:")
print(df["sentiment"].value_counts())