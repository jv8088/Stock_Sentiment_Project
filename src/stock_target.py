import pandas as pd

df = pd.read_csv("data/tesla_stock.csv")

print(df.head())

df["Target"] = (
    df["Close"].shift(-1) > df["Close"]
).astype(int)

print(
    df[["Close", "Target"]].head(10)
)

print("\nTarget Counts:")
print(df["Target"].value_counts())