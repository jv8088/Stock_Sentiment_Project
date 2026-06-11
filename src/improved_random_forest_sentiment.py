import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load Tesla data

df = pd.read_csv("data/tesla_stock.csv")

# Remove Yahoo Finance header rows

df = df.iloc[2:].copy()

# Convert numeric columns

cols = [
    "Open",
    "High",
    "Low",
    "Close",
    "Volume"
]

for col in cols:
    df[col] = pd.to_numeric(
        df[col],
        errors="coerce"
    )

df = df.dropna()

# -----------------------
# Technical Indicators
# -----------------------

df["MA20"] = (
    df["Close"]
    .rolling(20)
    .mean()
)

df["Range"] = (
    df["High"] - df["Low"]
)

df["Daily_Return"] = (
    df["Close"].pct_change()
)

df["Volatility"] = (
    df["Close"]
    .rolling(20)
    .std()
)

# -----------------------
# NEW SENTIMENT FEATURE
# -----------------------

np.random.seed(42)

df["Sentiment_Score"] = np.random.uniform(
    0,
    1,
    len(df)
)

# -----------------------
# Target
# -----------------------

df["Target"] = (
    df["Close"]
    .shift(-1)
    > df["Close"]
).astype(int)

df = df.dropna()

# -----------------------
# Features
# -----------------------

features = [

    "Open",
    "High",
    "Low",
    "Close",
    "Volume",

    "MA20",
    "Range",
    "Daily_Return",
    "Volatility",

    "Sentiment_Score"
]

X = df[features]

y = df["Target"]

# -----------------------
# Train Test Split
# -----------------------

X_train, X_test, y_train, y_test = train_test_split(

    X,
    y,

    test_size=0.2,

    random_state=42
)

# -----------------------
# Random Forest
# -----------------------

model = RandomForestClassifier(

    n_estimators=500,

    max_depth=10,

    random_state=42
)

model.fit(
    X_train,
    y_train
)

pred = model.predict(
    X_test
)

accuracy = accuracy_score(
    y_test,
    pred
)

print(
    "\nAccuracy:",
    accuracy
)