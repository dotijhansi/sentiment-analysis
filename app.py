import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import pipeline
from wordcloud import WordCloud

# Load Sentiment Analysis Models
default_model = "distilbert-base-uncased-finetuned-sst-2-english"
model_options = {
    "DistilBERT (Fast & Lightweight)": "distilbert-base-uncased-finetuned-sst-2-english",
    "BERT (Accurate but Slower)": "nlptown/bert-base-multilingual-uncased-sentiment",
    "RoBERTa (High Accuracy)": "cardiffnlp/twitter-roberta-base-sentiment"
}

st.sidebar.title("⚙️ Settings")
selected_model = st.sidebar.selectbox("Choose a Sentiment Model", list(model_options.keys()))

# Load the selected model
sentiment_pipeline = pipeline("sentiment-analysis", model=model_options[selected_model])

# UI Design
st.title("📝 Sentiment Analysis Web App")
st.write("Enter text below to analyze sentiment.")

# User Input
user_input = st.text_area("Enter text here:", "")

# Batch Processing Option
batch_input = st.file_uploader("Upload a Text File for Batch Analysis (Optional)", type=["txt"])

def visualize_sentiment(sentiment_scores):
    labels = list(sentiment_scores.keys())
    values = list(sentiment_scores.values())

    plt.figure(figsize=(6, 3))
    sns.barplot(x=labels, y=values, palette=["red", "gray", "green"])
    plt.ylim(0, 100)
    plt.title("Sentiment Confidence Scores")
    plt.xlabel("Sentiment")
    plt.ylabel("Confidence (%)")
    st.pyplot(plt)

def generate_wordcloud(text):
    wordcloud = WordCloud(width=800, height=400, background_color="white").generate(text)
    plt.figure(figsize=(8, 4))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    st.pyplot(plt)

if st.button("Analyze Sentiment"):
    if user_input:
        result = sentiment_pipeline(user_input)[0]
        label = result["label"]
        score = round(result["score"] * 100, 2)

        # Mapping Hugging Face output to a more intuitive label
        if "NEGATIVE" in label or "1 star" in label:
            sentiment = "Negative 😞"
            sentiment_scores = {"Negative": score, "Neutral": 100 - score, "Positive": 0}
            st.error(f"**{sentiment} ({score}%)**")
        elif "POSITIVE" in label or "5 stars" in label:
            sentiment = "Positive 😊"
            sentiment_scores = {"Negative": 0, "Neutral": 100 - score, "Positive": score}
            st.success(f"**{sentiment} ({score}%)**")
        else:
            sentiment = "Neutral 😐"
            sentiment_scores = {"Negative": 0, "Neutral": score, "Positive": 100 - score}
            st.info(f"**{sentiment} ({score}%)**")

        # Confidence Score Visualization
        visualize_sentiment(sentiment_scores)

        # Word Cloud
        generate_wordcloud(user_input)

    else:
        st.warning("Please enter text to analyze.")

# Batch Processing
if batch_input is not None:
    text_data = batch_input.read().decode("utf-8")
    st.subheader("📄 Batch Sentiment Analysis Results")

    sentences = text_data.split("\n")
    results = [sentiment_pipeline(sentence)[0] for sentence in sentences if sentence.strip()]

    for sentence, result in zip(sentences, results):
        label = result["label"]
        score = round(result["score"] * 100, 2)
        st.write(f"**Text:** {sentence}")
        st.write(f"📌 **Sentiment:** {label} ({score}%)")
        st.write("---")
