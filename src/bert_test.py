from transformers import pipeline

# Load BERT sentiment model
classifier = pipeline("sentiment-analysis")

tweets = [
    "Tesla is doing amazing this year",
    "I hate this company",
    "The stock market crashed badly today"
]

results = classifier(tweets)

for tweet, result in zip(tweets, results):
    print(f"\nTweet: {tweet}")
    print(result)