import pandas as pd
import random

positive_phrases = [
    "excellent quality",
    "highly recommended",
    "amazing product",
    "great battery life",
    "worth every penny",
    "fantastic performance",
    "superb build quality",
    "very satisfied",
    "outstanding experience",
    "works perfectly"
]

negative_phrases = [
    "waste of money",
    "poor quality",
    "very disappointed",
    "terrible product",
    "not worth the price",
    "bad battery life",
    "stopped working",
    "worst purchase ever",
    "extremely dissatisfied",
    "horrible experience"
]

neutral_phrases = [
    "average product",
    "works as expected",
    "nothing special",
    "decent quality",
    "okay performance",
    "acceptable results",
    "meets expectations",
    "reasonable purchase",
    "satisfactory quality",
    "moderate performance"
]

products = [
    "phone",
    "laptop",
    "tablet",
    "headphones",
    "speaker",
    "smartwatch",
    "camera",
    "monitor",
    "keyboard",
    "printer"
]

features = [
    "battery",
    "screen",
    "design",
    "performance",
    "quality",
    "delivery",
    "price",
    "service"
]

reviews = []

# --------------------
# Positive Reviews
# --------------------
for _ in range(4000):

    if random.random() < 0.30:
        review = (
            f"The {random.choice(features)} is average but overall "
            f"I am very satisfied with this {random.choice(products)}."
        )
    else:
        review = (
            f"This {random.choice(products)} has "
            f"{random.choice(positive_phrases)}."
        )

    reviews.append([review, "Positive"])

# --------------------
# Negative Reviews
# --------------------
for _ in range(4000):

    if random.random() < 0.30:
        review = (
            f"The {random.choice(features)} looks good but "
            f"the overall experience was terrible."
        )
    else:
        review = (
            f"This {random.choice(products)} has "
            f"{random.choice(negative_phrases)}."
        )

    reviews.append([review, "Negative"])

# --------------------
# Neutral Reviews
# --------------------
for _ in range(2000):

    neutral_templates = [
        f"The {random.choice(products)} is okay and works as expected.",
        f"The {random.choice(features)} is good but could be better.",
        f"This {random.choice(products)} offers decent value for money.",
        f"The product performs reasonably well.",
        f"The quality is acceptable for the price.",
        f"I have mixed feelings about this product."
    ]

    reviews.append(
        [random.choice(neutral_templates), "Neutral"]
    )

# Shuffle
random.shuffle(reviews)

df = pd.DataFrame(
    reviews,
    columns=["review", "sentiment"]
)
# --------------------
# Add Label Noise
# --------------------

noise_fraction = 0.10  # 10%

noise_indices = df.sample(
    frac=noise_fraction,
    random_state=42
).index

for idx in noise_indices:

    current_label = df.loc[idx, "sentiment"]

    possible_labels = [
        "Positive",
        "Negative",
        "Neutral"
    ]

    possible_labels.remove(current_label)

    df.loc[idx, "sentiment"] = random.choice(
        possible_labels
    )
df.to_csv(
    "synthetic_reviews.csv",
    index=False
)

print("Dataset Created Successfully")
print(df.shape)
print(df.head())