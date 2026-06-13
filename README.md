# 🛒 Product Review Sentiment Analyzer Using NLP

## 📌 Overview

The Product Review Sentiment Analyzer is a Natural Language Processing (NLP) based machine learning application that classifies product reviews into Positive, Negative, and Neutral sentiments.

The project analyzes customer reviews collected from Amazon and a synthetic review dataset to understand customer opinions and provide automated sentiment classification.

The application uses TF-IDF vectorization and a Linear Support Vector Classifier (LinearSVC) to predict sentiment from review text.

---

## 🎯 Objectives

- Analyze customer reviews automatically
- Classify reviews as Positive, Negative, or Neutral
- Visualize sentiment distribution
- Build a machine learning model using NLP techniques
- Provide real-time sentiment prediction through a Streamlit dashboard

---

## 🛠 Technologies Used

- Python
- Pandas
- NumPy
- Scikit-Learn
- TF-IDF Vectorization
- LinearSVC
- Streamlit
- Plotly
- Joblib

---

## 📂 Project Structure

```text
Product-Review-Sentiment-Analyzer/

│
├── data/
│   ├── reviews.csv
│   ├── synthetic_reviews.csv
│   └── final_reviews.csv
│
├── models/
│   └── sentiment_model.pkl
│
├── generate_dataset.py
├── train_model.py
├── app.py
├── requirements.txt
├── README.md
│
└── Product_Review_Sentiment_Analyzer_Report.pdf
```

---

## 📊 Dataset

The final dataset was created by combining:

- Amazon Product Review Dataset
- Synthetic Review Dataset

### Final Dataset Statistics

| Category | Count |
|----------|----------|
| Positive Reviews | 4925 |
| Negative Reviews | 4409 |
| Neutral Reviews | 2263 |
| Total Reviews | 11597 |

---

## 🔍 Data Preprocessing

The following preprocessing techniques were applied:

- Lowercase conversion
- Removal of special characters
- Removal of URLs
- Whitespace normalization
- Text cleaning using Regular Expressions

---

## 🤖 Machine Learning Model

### Feature Extraction

TF-IDF Vectorizer

### Classification Algorithm

Linear Support Vector Classifier (LinearSVC)

### Sentiment Classes

- Positive
- Negative
- Neutral

---

## 📈 Model Performance

| Metric | Score |
|----------|----------|
| Accuracy | 87.97% |
| Precision | 88% |
| Recall | 87% |
| F1 Score | 88% |

### Classification Results

| Sentiment | Precision | Recall | F1-Score |
|------------|------------|------------|------------|
| Negative | 0.87 | 0.89 | 0.88 |
| Neutral | 0.88 | 0.81 | 0.85 |
| Positive | 0.89 | 0.90 | 0.89 |

---

## 💻 Features

- Sentiment Prediction
- Review Analysis
- Sentiment Distribution Visualization
- Sample Review Display
- Interactive Streamlit Dashboard

---

## ▶️ How to Run

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Train Model

```bash
python train_model.py
```

### Run Dashboard

```bash
streamlit run app.py
```

---

## 📸 Screenshots

- Dashboard Home Page
- Sentiment Distribution Chart
- Positive Review Prediction
- Negative Review Prediction
- Model Performance Output

---

## 🔮 Future Scope

- Deep Learning Based Sentiment Analysis
- BERT Transformer Integration
- Multi-Language Review Analysis
- Aspect-Based Sentiment Analysis
- Real-Time E-Commerce Review Monitoring

---

## 👨‍💻 Author

Aayush Kumar

Minor Project – AI/ML
