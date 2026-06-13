import pandas as pd
import joblib
import re

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

# ----------------------
# Load Amazon Dataset
# ----------------------

amazon = pd.read_csv(
    "data/reviews.csv"
)

def sentiment_label(rating):

    if rating >= 4:
        return "Positive"

    elif rating == 3:
        return "Neutral"

    else:
        return "Negative"

amazon["sentiment"] = (
    amazon["reviews.rating"]
    .apply(sentiment_label)
)

amazon["review"] = (
    amazon["reviews.title"].fillna("")
    + " "
    +
    amazon["reviews.text"].fillna("")
)

amazon = amazon[
    ["review", "sentiment"]
]

# ----------------------
# Load Synthetic Dataset
# ----------------------

synthetic = pd.read_csv(
    "data/synthetic_reviews.csv"
)

# ----------------------
# Merge Datasets
# ----------------------

df = pd.concat(
    [amazon, synthetic],
    ignore_index=True
)

print("\nDataset Shape:")
print(df.shape)

print("\nClass Distribution:")
print(
    df["sentiment"]
    .value_counts()
)

# ----------------------
# Clean Text
# ----------------------

def clean_text(text):

    text = str(text).lower()

    text = re.sub(
        r"http\S+",
        "",
        text
    )

    text = re.sub(
        r"[^a-zA-Z\s]",
        " ",
        text
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()

df["review"] = (
    df["review"]
    .apply(clean_text)
)

# ----------------------
# Features & Labels
# ----------------------

X = df["review"]

y = df["sentiment"]

# ----------------------
# Train Test Split
# ----------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ----------------------
# Model Pipeline
# ----------------------

model = Pipeline([
    (
        "tfidf",
        TfidfVectorizer(
            stop_words="english",
            max_features=30000,
            ngram_range=(1,2),
            min_df=2,
            max_df=0.95,
            sublinear_tf=True
        )
    ),
    (
        "clf",
        LinearSVC(
            class_weight="balanced"
        )
    )
])

# ----------------------
# Train Model
# ----------------------

model.fit(
    X_train,
    y_train
)

# ----------------------
# Predictions
# ----------------------

pred = model.predict(
    X_test
)

# ----------------------
# Evaluation
# ----------------------

accuracy = accuracy_score(
    y_test,
    pred
)

print(
    f"\nAccuracy: {accuracy:.4f}"
)

print(
    classification_report(
        y_test,
        pred
    )
)

# ----------------------
# Save Model
# ----------------------

joblib.dump(
    model,
    "models/sentiment_model.pkl"
)

print(
    "\nModel Saved Successfully"
)