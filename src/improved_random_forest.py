import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Load stock data
df = pd.read_csv("data/tesla_stock.csv")

# Create target
df["Target"] = (
    df["Close"].shift(-1) > df["Close"]
).astype(int)

# Technical Indicators

# 5 Day Moving Average
df["MA5"] = df["Close"].rolling(5).mean()

# 10 Day Moving Average
df["MA10"] = df["Close"].rolling(10).mean()

# Daily Return
df["Daily_Return"] = df["Close"].pct_change()

# Volatility
df["Volatility"] = df["Daily_Return"].rolling(5).std()

# Drop missing rows
df = df.dropna()

# Features
X = df[
[
    "Open",
    "High",
    "Low",
    "Close",
    "Volume",
    "MA5",
    "MA10",
    "Daily_Return",
    "Volatility"
]
]

# Target
y = df["Target"]

# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Bigger Forest
model = RandomForestClassifier(
    n_estimators=500,
    max_depth=10,
    random_state=42
)

model.fit(X_train, y_train)

predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)

print("Accuracy:", accuracy)

print("\nClassification Report:")
print(classification_report(y_test, predictions))