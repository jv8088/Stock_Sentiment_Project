import pandas as pd

# Sample sentiment scores

sentiment_df = pd.DataFrame({

    "Date":[
        "2023-01-03",
        "2023-01-04",
        "2023-01-05",
        "2023-01-06",
        "2023-01-09"
    ],

    "Sentiment_Score":[
        0.82,
        0.74,
        0.45,
        0.65,
        0.91
    ]
})

print(sentiment_df)