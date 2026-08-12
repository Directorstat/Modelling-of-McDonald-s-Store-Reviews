import os
import streamlit as st
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords, wordnet
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag
from gensim import corpora
from gensim.models import LdaModel
import pickle
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import joblib # New import for joblib

# Ensure NLTK resources are downloaded
# These checks prevent re-downloading if already available
# Ensure required NLTK resources are available
try:
    nltk.data.find("corpora/stopwords")
except LookupError:
    nltk.download("stopwords", quiet=True)

try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt", quiet=True)
try:
    nltk.data.find("corpora/wordnet")
except LookupError:
    nltk.download("wordnet", quiet=True)
try:
    nltk.data.find('taggers/averaged_perceptron_tagger')
except nltk.downloader.DownloadError:
    nltk.download('averaged_perceptron_tagger')
try:
    nltk.data.find('sentiment/vader_lexicon')
except nltk.downloader.DownloadError:
    nltk.download('vader_lexicon')
stop_words = set(stopwords.words("english"))

# Initialize VADER sentiment analyzer
analyzer = SentimentIntensityAnalyzer()

# Load model artifacts
@st.cache_resource
def load_model_artifacts():
    lda_model = joblib.load("lda_model.joblib") # Load with joblib
    dictionary = joblib.load("lda_dictionary.joblib") # Load with joblib
    topic_names = joblib.load("topic_names.joblib") # Load with joblib
    return lda_model, dictionary, topic_names

lda_model, dictionary, topic_names = load_model_artifacts()

# Preprocessing functions (re-using the logic from the notebook)
stop_words = set(stopwords.words("english"))
stop_words.update(["mcdonalds", "mcdonald", "get", "got", "go", "went", "one", "us", "would"])
lemmatizer = WordNetLemmatizer()

def get_wordnet_pos(tag):
    if tag.startswith("J"):
        return wordnet.ADJ
    elif tag.startswith("V"):
        return wordnet.VERB
    elif tag.startswith("N"):
        return wordnet.NOUN
    elif tag.startswith("R"):
        return wordnet.ADV
    else:
        return wordnet.NOUN

def preprocess_text(text):
    text = str(text).lower() # 1. Lowercase
    text = re.sub(r"[^a-z\s]", " ", text) # 2. Remove punctuation and numbers
    tokens = word_tokenize(text) # 3. Tokenize
    tokens = [t for t in tokens if t not in stop_words and len(t) > 2] # 4. Remove stopwords & short words
    tagged_tokens = pos_tag(tokens)
    tokens = [lemmatizer.lemmatize(word, get_wordnet_pos(tag)) for word, tag in tagged_tokens] # 5. Lemmatise
    return tokens

def predict_dominant_topic(text):
    processed_tokens = preprocess_text(text)
    if not processed_tokens:
        return "No discernible topic (review too short or empty after preprocessing)", 0.0, []

    bow = dictionary.doc2bow(processed_tokens)
    topic_probs = lda_model.get_document_topics(bow)

    if not topic_probs:
        return "No discernible topic (words not in dictionary)", 0.0, []

    dominant_topic_id, dominant_prob = max(topic_probs, key=lambda x: x[1])
    dominant_topic_name = topic_names.get(dominant_topic_id, f"Topic {dominant_topic_id}")

    # Get top words for the dominant topic
    top_words_raw = lda_model.show_topic(dominant_topic_id, topn=10)
    top_words = [word for word, prob in top_words_raw]

    return dominant_topic_name, dominant_prob, top_words

# New function for review sentiment
def get_review_sentiment(text):
    vs = analyzer.polarity_scores(text)
    if vs['compound'] >= 0.05:
        label = "Positive"
        color = "#2e7d32" # Green
    elif vs['compound'] <= -0.05:
        label = "Negative"
        color = "#c62828" # Red
    else:
        label = "Neutral"
        color = "#ef6c00" # Orange
    return label, color, vs

# Topic sentiment mapping based on notebook's topic_summary analysis
topic_sentiment_mapping = {
    "Drive-Thru & Wait Times": {"label": "Negative", "color": "#c62828"}, # avg_rating 2.27
    "Order Delays & Follow-Up": {"label": "Neutral", "color": "#ef6c00"}, # avg_rating 2.58
    "Rude Staff & Poor Customer Service": {"label": "Negative", "color": "#c62828"}, # avg_rating 1.54
    "Order Accuracy (Wrong / Missing Items)": {"label": "Negative", "color": "#c62828"}, # avg_rating 2.42
    "Food & Menu Items": {"label": "Neutral", "color": "#ef6c00"}, # avg_rating 2.79
    "Cleanliness & Facility Complaints": {"label": "Neutral", "color": "#ef6c00"}, # avg_rating 3.44
    "Fast, Clean, Friendly Service (Positive)": {"label": "Positive", "color": "#2e7d32"}, # avg_rating 4.18
}


# Streamlit UI
st.set_page_config(page_title="McDonald's Review Topic Analyzer", page_icon=":hamburger:", layout="wide")

st.title(":hamburger: McDonald's Review Topic Analyzer")
st.markdown("Understand customer sentiment by identifying the main themes in McDonald's reviews using Latent Dirichlet Allocation.")

st.subheader("Enter a Customer Review")
review_input = st.text_area("", "The service was very slow and the fries were cold.", height=150)

if st.button("Analyze Review"):                  
    if review_input:
        dominant_topic_name, dominant_prob, top_words = predict_dominant_topic(review_input)
        review_sentiment_label, review_sentiment_color, sentiment_scores = get_review_sentiment(review_input)
        
        st.markdown("--- ")
        st.subheader("Analysis Result")
        
        # Display Review Sentiment
        st.write(f"#### Review Sentiment: <span style='color:{review_sentiment_color}'>**{review_sentiment_label}**</span>", unsafe_allow_html=True)
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric(label="Compound", value=f"{sentiment_scores['compound']:.2f}")
        with col2:
            st.metric(label="Positive", value=f"{sentiment_scores['pos']:.2f}")
        with col3:
            st.metric(label="Neutral", value=f"{sentiment_scores['neu']:.2f}")
        with col4:
            st.metric(label="Negative", value=f"{sentiment_scores['neg']:.2f}")
        
        st.markdown("") # Add some space

        if dominant_prob > 0.0:
            st.success(f"**Dominant Topic:** {dominant_topic_name}")
            st.info(f"**Confidence:** {dominant_prob:.2f}")
            st.write("**Key words for this topic:** ", ", ".join(top_words))
            
            # Display Topic Sentiment
            topic_info = topic_sentiment_mapping.get(dominant_topic_name, {"label": "Unknown", "color": "gray"})
            st.write(f"#### Implied Topic Sentiment: <span style='color:{topic_info['color']}'>**{topic_info['label']}**</span>", unsafe_allow_html=True)
            
            st.markdown("### Interpretation")
            if topic_info['label'] == "Negative":
                st.warning(f"This review strongly aligns with a negative theme related to {dominant_topic_name.lower()}. This indicates areas for improvement.")
            elif topic_info['label'] == "Positive":
                st.success(f"This is a positive review, aligning with themes of {dominant_topic_name.lower()}. Great work!")
            else: # Neutral
                st.info(f"This review addresses a neutral-to-mildly-negative theme related to {dominant_topic_name.lower()}. If the sentiment is overall positive, it indicates an acceptable experience within this category.")
        else:
            st.error(dominant_topic_name)
    else:
        st.warning("Please enter a review to analyze.")

st.markdown("--- ")
st.markdown("**How this works:** This app uses a Latent Dirichlet Allocation (LDA) model to categorize customer reviews into 7 predefined topics. The model analyzes the words in your input review to determine which topic it most strongly belongs to. Sentiment analysis is performed on the review text to provide a direct sentiment score.")
