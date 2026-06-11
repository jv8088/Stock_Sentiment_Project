from transformers import pipeline
import pandas as pd

classifier = pipeline("sentiment-analysis")

tweets = [

    "Tesla beats earnings expectations",

    "Tesla opens a new gigafactory",

    "Tesla stock crashes after poor results",

    "Tesla delivery numbers disappoint investors",

    "Tesla AI event excites shareholders"

]

scores = []

for tweet in tweets:

    result = classifier(tweet)[0]

    if result["label"] == "POSITIVE":

        score = result["score"]

    else:

        score = 1 - result["score"]

    scores.append(score)

df = pd.DataFrame({

    "Tweet": tweets,

    "Sentiment_Score": scores

})

print(df)