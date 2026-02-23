import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# -----------------------------
# Title
# -----------------------------
st.title("🎬 IMDB Movie Review Sentiment Analysis")
st.write("""
This app predicts whether a movie review is **Positive** or **Negative** using a Naive Bayes model.
""")

# -----------------------------
# Upload Dataset
# -----------------------------
st.sidebar.header("Upload IMDB Dataset CSV")
uploaded_file = st.sidebar.file_uploader("Choose CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("Dataset Preview:", df.head())

    # -----------------------------
    # Preprocessing
    # -----------------------------
    df['sentiment'] = df['sentiment'].map({'positive': 1, 'negative': 0})
    X = df['review']
    y = df['sentiment']

    # Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # TF-IDF Vectorization
    tfidf = TfidfVectorizer(stop_words="english", max_features=5000)
    X_train_tfidf = tfidf.fit_transform(X_train)
    X_test_tfidf = tfidf.transform(X_test)

    # Train Naive Bayes Model
    nb_model = MultinomialNB()
    nb_model.fit(X_train_tfidf, y_train)

    # Predictions and Evaluation
    y_pred = nb_model.predict(X_test_tfidf)
    accuracy = accuracy_score(y_test, y_pred)

    st.subheader("📊 Model Evaluation Metrics")
    st.write(f"**Accuracy:** {accuracy*100:.2f}%")
    st.text("Classification Report:")
    st.text(classification_report(y_test, y_pred))

    # Confusion Matrix Plot
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Negative", "Positive"],
                yticklabels=["Negative", "Positive"], ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix")
    st.pyplot(fig)

    # -----------------------------
    # User Input for Prediction
    # -----------------------------
    st.sidebar.header("Enter a Movie Review for Prediction")
    user_review = st.sidebar.text_area("Movie Review", "")

    if st.sidebar.button("Predict"):
        if user_review.strip() == "":
            st.warning("Please enter a review to predict sentiment.")
        else:
            review_tfidf = tfidf.transform([user_review])
            prediction = nb_model.predict(review_tfidf)[0]
            sentiment = "Positive ✅" if prediction == 1 else "Negative ❌"
            st.subheader(f"Prediction for the review:")
            st.write(sentiment)
else:
    st.info("Please upload an IMDB dataset CSV file to proceed.")
