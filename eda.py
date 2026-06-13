import pandas as pd

df = pd.read_csv(
    "data/reviews.csv"
)

print("Shape:")
print(df.shape)

print("\nColumns:")
print(df.columns.tolist())

print("\nRatings:")
print(
    df["reviews.rating"]
    .value_counts()
)

print("\nMissing Reviews:")
print(
    df["reviews.text"]
    .isna()
    .sum()
)

print("\nSample Reviews:")
print(
    df["reviews.text"]
    .head()
)