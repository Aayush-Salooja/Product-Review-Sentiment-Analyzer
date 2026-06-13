import pandas as pd

amazon = pd.read_csv("data/reviews.csv")

def sentiment_label(rating):
    if rating >= 4:
        return "Positive"
    elif rating == 3:
        return "Neutral"
    else:
        return "Negative"

amazon["sentiment"] = amazon["reviews.rating"].apply(sentiment_label)

amazon["review"] = (
    amazon["reviews.title"].fillna("") +
    " " +
    amazon["reviews.text"].fillna("")
)

amazon = amazon[["review", "sentiment"]]

synthetic = pd.read_csv("data/synthetic_reviews.csv")

final_df = pd.concat(
    [amazon, synthetic],
    ignore_index=True
)

final_df.to_csv(
    "data/final_reviews.csv",
    index=False
)

print(final_df.columns)
print(final_df.head())