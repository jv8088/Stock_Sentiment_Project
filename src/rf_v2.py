import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load data
df = pd.read_csv("data/tesla_stock.csv")

# Target variable
df["Target"] = (
    df["Close"].shift(-1) > df["Close"]
).astype(int)

# Technical Indicators

# Moving Averages
df["MA5"] = df["Close"].rolling(5).mean()
df["MA10"] = df["Close"].rolling(10).mean()
df["MA20"] = df["Close"].rolling(20).mean()

# Exponential Moving Average
df["EMA10"] = df["Close"].ewm(span=10).mean()

# Daily Return
df["Daily_Return"] = df["Close"].pct_change()

# Volatility
df["Volatility"] = df["Daily_Return"].rolling(5).std()

# Price Range
df["Range"] = df["High"] - df["Low"]

# Momentum
df["Momentum"] = df["Close"] - df["Close"].shift(5)

# Remove NaN rows
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
    "MA20",
    "EMA10",
    "Daily_Return",
    "Volatility",
    "Range",
    "Momentum"
]
]

# Target
y = df["Target"]

# Time-series split
split = int(len(df) * 0.8)

X_train = X[:split]
X_test = X[split:]

y_train = y[:split]
y_test = y[split:]

# Random Forest
model = RandomForestClassifier(
    n_estimators=1000,
    max_depth=15,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42
)

model.fit(X_train, y_train)

predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)

print("\nAccuracy:", accuracy)

print("\nClassification Report:")
print(classification_report(y_test, predictions))

# Feature Importance
importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": model.feature_importances_
})

print("\nFeature Importance:")
print(
    importance.sort_values(
        by="Importance",
        ascending=False
    )
)