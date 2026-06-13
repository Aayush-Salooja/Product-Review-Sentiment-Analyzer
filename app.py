import streamlit as st
import pandas as pd
import joblib
import plotly.express as px

# ----------------------
# Page Config
# ----------------------

st.set_page_config(
    page_title="Product Review Sentiment Analyzer",
    page_icon="🛒",
    layout="wide"
)

# ----------------------
# Load Data
# ----------------------

df = pd.read_csv(
    "data/final_reviews.csv"
)

model = joblib.load(
    "models/sentiment_model.pkl"
)

# ----------------------
# Title
# ----------------------

st.title("🛒 Product Review Sentiment Analyzer")

st.markdown(
    """
    Analyze product reviews using NLP and Machine Learning.
    """
)

# ----------------------
# Dataset Stats
# ----------------------

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "Total Reviews",
        len(df)
    )

with col2:
    st.metric(
        "Unique Sentiments",
        df["sentiment"].nunique()
    )

with col3:
    st.metric(
        "Model Accuracy",
        "87.97%"
    )

# ----------------------
# Sentiment Distribution
# ----------------------

st.subheader("📊 Sentiment Distribution")

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
    fig,
    use_container_width=True
)

# ----------------------
# Review Length Analysis
# ----------------------

st.subheader("📈 Review Length Analysis")

df["review_length"] = (
    df["review"]
    .astype(str)
    .apply(len)
)

fig2 = px.histogram(
    df,
    x="review_length",
    nbins=30,
    title="Distribution of Review Lengths"
)

st.plotly_chart(
    fig2,
    use_container_width=True
)

# ----------------------
# Sentiment Prediction
# ----------------------

st.subheader("🤖 Predict Review Sentiment")

review = st.text_area(
    "Enter a product review"
)

if st.button("Analyze Sentiment"):

    if review.strip() == "":
        st.warning(
            "Please enter a review."
        )

    else:

        prediction = model.predict(
            [review]
        )[0]

        if prediction == "Positive":

            st.success(
                f"Predicted Sentiment: {prediction} 😊"
            )

        elif prediction == "Negative":

            st.error(
                f"Predicted Sentiment: {prediction} 😞"
            )

        else:

            st.warning(
                f"Predicted Sentiment: {prediction} 😐"
            )

# ----------------------
# Sample Reviews
# ----------------------

st.subheader("📝 Sample Reviews")

st.dataframe(
    df.sample(10),
    use_container_width=True
)