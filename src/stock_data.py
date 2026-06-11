import yfinance as yf

stock = yf.download(
    "TSLA",
    start="2023-01-01",
    end="2024-01-01",
    auto_adjust=False
)

stock.columns = stock.columns.get_level_values(0)

stock.to_csv("data/tesla_stock.csv")

print(stock.head())
print("\nShape:", stock.shape)