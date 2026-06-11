import pandas as pd

# Load Tesla stock data

stock = pd.read_csv("data/tesla_stock.csv")

# Remove Yahoo Finance extra rows

stock = stock.iloc[2:].copy()

# Keep only first 5 rows for testing

stock = stock.head(5)

# Create sentiment scores

sentiment_df = pd.DataFrame({

    "Sentiment_Score":[
        0.82,
        0.74,
        0.45,
        0.65,
        0.91
    ]
})

# Merge

stock["Sentiment_Score"] = sentiment_df["Sentiment_Score"]

print(
    stock[
        ["Close","Volume","Sentiment_Score"]
    ]
)