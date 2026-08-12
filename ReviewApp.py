import os
import streamlit as st
import pandas as pd
import re
import nltk

from nltk.corpus import stopwords, wordnet
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag
from nltk.sentiment.vader import SentimentIntensityAnalyzer

from gensim import corpora
from gensim.models import LdaModel

import pickle
import joblib


# ============================================================
# STREAMLIT PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="McDonald's Review Topic Analyzer",
    page_icon="🍔",
    layout="wide"
)


# ============================================================
# NLTK RESOURCE CONFIGURATION
# ============================================================

# Store NLTK resources inside the application directory.
# This makes deployment on Streamlit Cloud more reliable.

NLTK_DATA_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "nltk_data"
)

os.makedirs(NLTK_DATA_DIR, exist_ok=True)

# Tell NLTK where to look for downloaded resources.
if NLTK_DATA_DIR not in nltk.data.path:
    nltk.data.path.append(NLTK_DATA_DIR)


def download_nltk_resource(resource_path, resource_name):
    """
    Check whether an NLTK resource exists.
    Download it if it is missing.
    """

    try:
        nltk.data.find(resource_path)

    except LookupError:
        try:
            nltk.download(
                resource_name,
                download_dir=NLTK_DATA_DIR,
                quiet=True
            )
        except Exception as e:
            st.error(
                f"Unable to download required NLTK resource: "
                f"{resource_name}. Error: {e}"
            )
            st.stop()


# Required NLTK resources

download_nltk_resource(
    "corpora/stopwords",
    "stopwords"
)

download_nltk_resource(
    "tokenizers/punkt",
    "punkt"
)

download_nltk_resource(
    "tokenizers/punkt_tab",
    "punkt_tab"
)

download_nltk_resource(
    "corpora/wordnet",
    "wordnet"
)

download_nltk_resource(
    "taggers/averaged_perceptron_tagger",
    "averaged_perceptron_tagger"
)

# Newer NLTK versions use this resource for POS tagging.
download_nltk_resource(
    "taggers/averaged_perceptron_tagger_eng",
    "averaged_perceptron_tagger_eng"
)

download_nltk_resource(
    "sentiment/vader_lexicon",
    "vader_lexicon"
)


# ============================================================
# STOPWORDS
# ============================================================

stop_words = set(stopwords.words("english"))

stop_words.update([
    "mcdonalds",
    "mcdonald",
    "get",
    "got",
    "go",
    "went",
    "one",
    "us",
    "would"
])


# ============================================================
# INITIALIZE NLP COMPONENTS
# ============================================================

lemmatizer = WordNetLemmatizer()

analyzer = SentimentIntensityAnalyzer()


# ============================================================
# LOAD MODEL ARTIFACTS
# ============================================================

@st.cache_resource
def load_model_artifacts():

    lda_model = joblib.load(
        "lda_model.joblib"
    )

    dictionary = joblib.load(
        "lda_dictionary.joblib"
    )

    topic_names = joblib.load(
        "topic_names.joblib"
    )

    return lda_model, dictionary, topic_names


# Load the trained LDA model
lda_model, dictionary, topic_names = load_model_artifacts()


# ============================================================
# WORDNET POS MAPPING
# ============================================================

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


# ============================================================
# TEXT PREPROCESSING
# ============================================================

def preprocess_text(text):

    # 1. Convert text to lowercase
    text = str(text).lower()

    # 2. Remove punctuation and numbers
    text = re.sub(
        r"[^a-z\s]",
        " ",
        text
    )

    # 3. Tokenize
    tokens = word_tokenize(text)

    # 4. Remove stopwords and short words
    tokens = [
        token
        for token in tokens
        if token not in stop_words
        and len(token) > 2
    ]

    # 5. POS tagging
    tagged_tokens = pos_tag(tokens)

    # 6. Lemmatization
    tokens = [
        lemmatizer.lemmatize(
            word,
            get_wordnet_pos(tag)
        )
        for word, tag in tagged_tokens
    ]

    return tokens


# ============================================================
# LDA TOPIC PREDICTION
# ============================================================

def predict_dominant_topic(text):

    processed_tokens = preprocess_text(text)

    # No usable words
    if not processed_tokens:

        return (
            "No discernible topic "
            "(review too short or empty after preprocessing)",
            0.0,
            []
        )

    # Convert processed text to Bag of Words
    bow = dictionary.doc2bow(
        processed_tokens
    )

    # Get topic probabilities
    topic_probs = lda_model.get_document_topics(
        bow
    )

    # No matching words in dictionary
    if not topic_probs:

        return (
            "No discernible topic "
            "(words not in dictionary)",
            0.0,
            []
        )

    # Identify dominant topic
    dominant_topic_id, dominant_prob = max(
        topic_probs,
        key=lambda x: x[1]
    )

    # Get human-readable topic name
    dominant_topic_name = topic_names.get(
        dominant_topic_id,
        f"Topic {dominant_topic_id}"
    )

    # Get top words for the dominant topic
    top_words_raw = lda_model.show_topic(
        dominant_topic_id,
        topn=10
    )

    top_words = [
        word
        for word, probability in top_words_raw
    ]

    return (
        dominant_topic_name,
        dominant_prob,
        top_words
    )


# ============================================================
# REVIEW SENTIMENT ANALYSIS
# ============================================================

def get_review_sentiment(text):

    sentiment_scores = analyzer.polarity_scores(
        text
    )

    compound = sentiment_scores["compound"]

    if compound >= 0.05:

        label = "Positive"
        color = "#2e7d32"

    elif compound <= -0.05:

        label = "Negative"
        color = "#c62828"

    else:

        label = "Neutral"
        color = "#ef6c00"

    return (
        label,
        color,
        sentiment_scores
    )


# ============================================================
# TOPIC SENTIMENT MAPPING
# ============================================================

topic_sentiment_mapping = {

    "Drive-Thru & Wait Times": {
        "label": "Negative",
        "color": "#c62828"
    },

    "Order Delays & Follow-Up": {
        "label": "Neutral",
        "color": "#ef6c00"
    },

    "Rude Staff & Poor Customer Service": {
        "label": "Negative",
        "color": "#c62828"
    },

    "Order Accuracy (Wrong / Missing Items)": {
        "label": "Negative",
        "color": "#c62828"
    },

    "Food & Menu Items": {
        "label": "Neutral",
        "color": "#ef6c00"
    },

    "Cleanliness & Facility Complaints": {
        "label": "Neutral",
        "color": "#ef6c00"
    },

    "Fast, Clean, Friendly Service (Positive)": {
        "label": "Positive",
        "color": "#2e7d32"
    }
}


# ============================================================
# APPLICATION HEADER
# ============================================================

st.title(
    "🍔 McDonald's Review Topic Analyzer"
)

st.markdown(
    """
    Understand customer sentiment by identifying the main
    themes in McDonald's reviews using Latent Dirichlet
    Allocation (LDA).
    """
)


# ============================================================
# REVIEW INPUT
# ============================================================

st.subheader(
    "Enter a Customer Review"
)

review_input = st.text_area(
    "",
    "The service was very slow and the fries were cold.",
    height=150
)


# ============================================================
# ANALYZE BUTTON
# ============================================================

if st.button(
    "Analyze Review",
    type="primary"
):

    if review_input and review_input.strip():

        # ----------------------------------------------------
        # TOPIC ANALYSIS
        # ----------------------------------------------------

        (
            dominant_topic_name,
            dominant_prob,
            top_words
        ) = predict_dominant_topic(
            review_input
        )


        # ----------------------------------------------------
        # SENTIMENT ANALYSIS
        # ----------------------------------------------------

        (
            review_sentiment_label,
            review_sentiment_color,
            sentiment_scores
        ) = get_review_sentiment(
            review_input
        )


        st.markdown("---")

        st.subheader(
            "Analysis Result"
        )


        # ====================================================
        # REVIEW SENTIMENT
        # ====================================================

        st.markdown(
            f"""
            #### Review Sentiment:
            <span style="color:{review_sentiment_color};
                         font-weight:bold;">
                {review_sentiment_label}
            </span>
            """,
            unsafe_allow_html=True
        )


        # ====================================================
        # SENTIMENT METRICS
        # ====================================================

        col1, col2, col3, col4 = st.columns(4)

        with col1:

            st.metric(
                label="Compound",
                value=f"{sentiment_scores['compound']:.2f}"
            )

        with col2:

            st.metric(
                label="Positive",
                value=f"{sentiment_scores['pos']:.2f}"
            )

        with col3:

            st.metric(
                label="Neutral",
                value=f"{sentiment_scores['neu']:.2f}"
            )

        with col4:

            st.metric(
                label="Negative",
                value=f"{sentiment_scores['neg']:.2f}"
            )


        st.markdown("")


        # ====================================================
        # DOMINANT TOPIC
        # ====================================================

        if dominant_prob > 0.0:

            st.success(
                f"Dominant Topic: {dominant_topic_name}"
            )

            st.info(
                f"Confidence: {dominant_prob:.2f}"
            )


            # ------------------------------------------------
            # TOP WORDS
            # ------------------------------------------------

            st.write(
                "**Key words for this topic:**",
                ", ".join(top_words)
            )


            # =================================================
            # TOPIC SENTIMENT
            # =================================================

            topic_info = topic_sentiment_mapping.get(
                dominant_topic_name,
                {
                    "label": "Unknown",
                    "color": "gray"
                }
            )


            st.markdown(
                f"""
                #### Implied Topic Sentiment:
                <span style="color:{topic_info['color']};
                             font-weight:bold;">
                    {topic_info['label']}
                </span>
                """,
                unsafe_allow_html=True
            )


            # =================================================
            # INTERPRETATION
            # =================================================

            st.markdown(
                "### Interpretation"
            )


            if topic_info["label"] == "Negative":

                st.warning(
                    f"This review strongly aligns with a "
                    f"negative theme related to "
                    f"{dominant_topic_name.lower()}. "
                    f"This indicates areas for improvement."
                )


            elif topic_info["label"] == "Positive":

                st.success(
                    f"This is a positive review, aligning "
                    f"with themes of "
                    f"{dominant_topic_name.lower()}. "
                    f"Great work!"
                )


            else:

                st.info(
                    f"This review addresses a "
                    f"neutral-to-mildly-negative theme "
                    f"related to "
                    f"{dominant_topic_name.lower()}. "
                    f"If the sentiment is overall positive, "
                    f"it indicates an acceptable experience "
                    f"within this category."
                )


        else:

            st.error(
                dominant_topic_name
            )


    else:

        st.warning(
            "Please enter a review to analyze."
        )


# ============================================================
# HOW THE APP WORKS
# ============================================================

st.markdown("---")

st.markdown(
    """
    **How this works:**

    This app uses a Latent Dirichlet Allocation (LDA) model
    to categorize customer reviews into 7 predefined topics.

    The model analyzes the words in your input review to
    determine which topic it most strongly belongs to.

    Sentiment analysis is performed using VADER to provide
    positive, neutral, negative, and compound sentiment scores.
    """
)
