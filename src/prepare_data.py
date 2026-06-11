import pandas as pd
import re

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

def clean_tweet(text):

    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"#\w+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)

    return text.lower().strip()

df["clean_tweet"] = df["tweet"].apply(clean_tweet)

# Convert labels
df["sentiment"] = df["sentiment"].replace({
    4: 1
})

print(df[["sentiment", "clean_tweet"]].head())

print("\nSentiment Counts:")
print(df["sentiment"].value_counts())