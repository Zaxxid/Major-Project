import streamlit as st
import tweepy  # type: ignore
import pandas as pd  # type: ignore
import plotly.express as px  # type: ignore
import time
import re
from transformers import pipeline
from wordcloud import WordCloud
import matplotlib.pyplot as plt  # type: ignore
from collections import Counter
from datetime import datetime, timedelta, timezone
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import numpy as np
import os
from PIL import Image
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.service import Service 
# ==========================
# 🎓 Final Year Project Header
# ==========================
st.set_page_config(page_title="Final Year Project: Real-Time Twitter Sentiment Analyzer", layout="wide")
st.markdown("""
<style>
    .title {
        font-size: 50px;
        font-weight: bold;
        color: #1DA1F2;
        text-align: center;
    }
    .subtitle {
        font-size: 18px;
        text-align: center;
        margin-bottom: 10px;
    }
    .footer {
        font-size: 12px;
        text-align: center;
        margin-top: 50px;
        color: gray;
    }
    .stButton>button {
        background-color: #1DA1F2;
        color: white;
        border-radius: 8px;
        font-weight: bold;
    }
</style>
<div class="title">Real-Time Twitter Sentiment Analyzer: A Final Year Project</div>
<div class="subtitle">By Zaid Usmani , Nikhil Dash , Mohd Saad | Supervised by Dr.Manu Singh | ABES Engineering College</div>
""", unsafe_allow_html=True)


# ==========================
# Twitter API Credentials (Replace with your own keys)
# ==========================
BEARER_TOKEN ="AAAAAAAAAAAAAAAAAAAAABh51AEAAAAATrMq%2FyOvIyngTxEE19j%2FoYZv7u0%3DWmpdeDa8bz294v92xwt1VIPCOne3M64WHmBKDmeRfZCro1auvj"

# Authenticate Tweepy with Bearer Token
client = tweepy.Client(bearer_token=BEARER_TOKEN)

# Initialize Models
sentiment_tokenizer = AutoTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment")
sentiment_model = AutoModelForSequenceClassification.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment")
sarcasm_detector = pipeline("text-classification", model="mrm8488/t5-base-finetuned-sarcasm-twitter")

# ==========================
# Function to Analyze Sentiment
# ==========================
def analyze_sentiment(text):
    encoded_input = sentiment_tokenizer(text, return_tensors='pt', truncation=True)
    output = sentiment_model(**encoded_input)
    scores = torch.softmax(output.logits, dim=1).detach().numpy()[0]
    labels = ['Negative', 'Neutral', 'Positive']
    sentiment = labels[np.argmax(scores)]
    confidence = np.max(scores)
    return sentiment, confidence

# ==========================
# Search Form UI
# ==========================
st.sidebar.title("🔍 Analyze Tweets")
st.sidebar.markdown("Use the form below to input your query and time frame.")

with st.sidebar.form(key='search_form'):
    search_query = st.text_input('Enter search keyword/hashtag:', placeholder='e.g., AI')
    max_tweets = st.number_input('Number of tweets to analyze:', min_value=10, max_value=500, value=50)
    since_date = st.date_input("From Date:")
    until_date = st.date_input("Until Date:")
    exclude_words = st.text_input("Exclude tweets containing (comma-separated):", placeholder="spam, promo")
    confidence_threshold = st.slider("Minimum Sentiment Confidence:", 0.0, 1.0, 0.5, 0.05)
    real_time = st.checkbox("Enable Real-Time Streaming (Auto-refresh)")
    submit_button = st.form_submit_button(label='Analyze Tweets')

# ==========================
# Function to Fetch Tweets
# ==========================
def fetch_tweets(query, max_results, since, until, exclude_list):
    tweets_list = []
    query = f"{query} -is:retweet"
    start_time = f"{since}T00:00:00Z" if since else None
    end_time = f"{until}T23:59:59Z" if until else None

    if end_time:
        now = datetime.now(timezone.utc)
        adjusted_end_time = now - timedelta(seconds=600)
        end_time = adjusted_end_time.strftime('%Y-%m-%dT%H:%M:%SZ')

    for attempt in range(5):
        try:
            tweets = client.search_recent_tweets(query=query, max_results=max_results, tweet_fields=["created_at", "author_id", "public_metrics", "geo"], start_time=start_time, end_time=end_time)
            for tweet in tweets.data:
                tweet_text = tweet.text
                tweet_likes = tweet.public_metrics['like_count']
                tweet_quotes = tweet.public_metrics.get('quote_count', 0)
                tweet_time = tweet.created_at.strftime('%Y-%m-%d %H:%M:%S')
                tweet_retweet = tweet.public_metrics.get('retweet_count', 0)
                tweet_url = f"https://twitter.com/{tweet.author_id}/status/{tweet.id}"
                if any(word.lower() in tweet_text.lower() for word in exclude_list):
                    continue
                tweets_list.append({
                    'text': tweet_text,
                    'likes': tweet_likes,
                    'quotes': tweet_quotes,
                    'created_at': tweet_time,
                    'retweets': tweet_retweet,
                    'url': tweet_url
                })
            return tweets_list
        except tweepy.TooManyRequests:
            time.sleep(60)
        except Exception as e:
            print(f"Error: {e}")
            break
    return tweets_list

# ==========================
# Process and Display Results
# ==========================
if submit_button and search_query:
    with st.spinner('⏳ Fetching and analyzing tweets...'):
        exclude_list = [word.strip().lower() for word in exclude_words.split(',') if word.strip()]
        tweets = fetch_tweets(search_query, max_tweets, since_date, until_date, exclude_list)

        if not tweets:
            st.error("No tweets found.")
        else:
            df = pd.DataFrame(tweets)
            df[['Sentiment', 'Confidence']] = df['text'].apply(lambda x: pd.Series(analyze_sentiment(x)))
            sarcasm_label_mapping = {"LABEL_0": "Sarcasm", "LABEL_1": "Not Sarcasm"}
            df['Sarcasm'] = df['text'].apply(lambda x: sarcasm_label_mapping.get(sarcasm_detector(x)[0]['label'], "Unknown"))

            st.session_state.df = df
            if 'url' in df.columns:
                st.session_state.tweet_urls = df['url'].tolist()

            sentiment_counts = df['Sentiment'].str.upper().value_counts()
            cols = st.columns(4)
            metrics = {
                "😃 Positive": sentiment_counts.get('POSITIVE', 0),
                "😠 Negative": sentiment_counts.get('NEGATIVE', 0),
                "😐 Neutral": sentiment_counts.get('NEUTRAL', 0),
                "❤️ Total Likes": df['likes'].sum()
            }
            for col, (label, value) in zip(cols, metrics.items()):
                col.metric(label, f"{value:,}")

            # 📊 Charts
            st.subheader("📊 Visual Analysis")
            fig_sentiment = px.histogram(df, x='Sentiment', title='Sentiment Distribution', color='Sentiment')
            st.plotly_chart(fig_sentiment)

            df['created_at'] = pd.to_datetime(df['created_at'])
            fig_confidence = px.line(df, x='created_at', y='Confidence', title='Confidence Over Time')
            st.plotly_chart(fig_confidence)

            fig_sarcasm = px.histogram(df, x='Sarcasm', title='Sarcasm Detection Results', color='Sarcasm')
            st.plotly_chart(fig_sarcasm)

            # 📄 Data Table
            st.subheader("📄 Raw Tweet Data")
            st.dataframe(df[['text', 'likes', 'quotes', 'Sentiment', 'Confidence', 'created_at', 'retweets', 'url']])

            if 'df' in st.session_state:
                    csv = st.session_state.df.to_csv(index=False).encode('utf-8')
                    st.download_button("📥 Download CSV Report", csv, "tweet_analysis.csv", "text/csv")

# Function to capture screenshot
def capture_tweet_screenshot(tweet_url):
    chrome_options = Options()
    chrome_options.add_argument("--headless=new")
    chrome_options.add_argument("--window-size=1200,800")
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--no-sandbox")

    screenshot_image = None
    try:
        service = Service(ChromeDriverManager().install())
        driver = webdriver.Chrome(service=service, options=chrome_options)
        driver.get(tweet_url)
        time.sleep(5)
        screenshot_image = driver.get_screenshot_as_png()
    except Exception as e:
        st.error(f"Error capturing screenshot for: {tweet_url}")
        st.error(str(e))
    finally:
        driver.quit()

    return screenshot_image


# Display screenshots if URLs are available
if 'tweet_urls' in st.session_state:
    st.subheader("🖼️ Tweet Screenshots")
    for i, tweet_url in enumerate(st.session_state.tweet_urls):
        with st.spinner(f"Capturing screenshot for Tweet {i+1}..."):
            screenshot_image = capture_tweet_screenshot(tweet_url)
        if screenshot_image:
            st.image(screenshot_image, caption=f"Tweet Screenshot {i+1} - {tweet_url}", use_container_width=True)
        else:
            st.warning(f"Screenshot not available for Tweet {i+1} - {tweet_url}")

                
        
# ==========================
# 📘 Project Summary Panel
# ==========================
st.markdown("""
---
### 📘 Project Summary
- **Goal**: Real-time sentiment & sarcasm analysis of trending tweets
- **Dataset**: Twitter API v2 live tweets
- **Models**: CardiffNLP RoBERTa (Sentiment), mrm8488 T5 (Sarcasm)
- **Tech Stack**: Python, Streamlit, HuggingFace Transformers, Plotly, Twitter API
- **GitHub**: [View Repository](https://github.com/Zaxxid/Major-Project)
""")

st.markdown('<div class="footer">© 2025 Zaid Usmani , Nikhil Dash , Mohd Saad | Final Year Project | ABES Engineering College</div>', unsafe_allow_html=True)
