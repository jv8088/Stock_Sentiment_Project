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

    text = text.lower()

    return text.strip()

df["clean_tweet"] = df["tweet"].apply(clean_tweet)

print(df[["tweet", "clean_tweet"]].head())