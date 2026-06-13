import streamlit as st
import pandas as pd
import joblib
import plotly.express as px

# --------------------
# Load Data
# --------------------

df = pd.read_csv(
    "data/synthetic_reviews.csv"
)

model = joblib.load(
    "models/sentiment_model.pkl"
)

# --------------------
# Title
# --------------------

st.title(
    "🛒 Product Review Sentiment Analyzer"
)

st.write(
    "Analyze product reviews using NLP and Machine Learning."
)

# --------------------
# Dataset Overview
# --------------------

st.subheader(
    "Dataset Overview"
)

st.write(
    f"Total Reviews: {len(df)}"
)

# --------------------
# Sentiment Distribution
# --------------------

st.subheader(
    "Sentiment Distribution"
)

sentiment_counts = (
    df["sentiment"]
    .value_counts()
)

fig = px.pie(
    values=sentiment_counts.values,
    names=sentiment_counts.index,
    title="Review Sentiment Distribution"
)

st.plotly_chart(
    fig
)

# --------------------
# Prediction
# --------------------

st.subheader(
    "Predict Review Sentiment"
)

review = st.text_area(
    "Enter Review"
)

if st.button(
    "Analyze"
):

    prediction = model.predict(
        [review]
    )[0]

    if prediction == "Positive":
        st.success(
            f"Sentiment: {prediction} 😊"
        )

    elif prediction == "Negative":
        st.error(
            f"Sentiment: {prediction} 😞"
        )

    else:
        st.warning(
            f"Sentiment: {prediction} 😐"
        )

# --------------------
# Sample Reviews
# --------------------

st.subheader(
    "Sample Reviews"
)

st.dataframe(
    df.sample(10)
)