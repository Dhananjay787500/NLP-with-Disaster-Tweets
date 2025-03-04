import streamlit as st
import joblib
import numpy as np
import re
import string
import requests
from datetime import datetime, timedelta
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Load the trained model and vectorizers
best_rf_model = joblib.load('best_rf_model.pkl')
tfidf = joblib.load('tfidf_vectorizer.pkl')
w2v_model = joblib.load('w2v_model.pkl')

# Initialize NLTK components
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# Function to preprocess new text data
def preprocess_new_text(text):
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\d+', '', text)
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stop_words]
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return ' '.join(tokens), tokens

# Function to get TF-IDF and Word2Vec features
def get_features(text):
    cleaned_text, tokens = preprocess_new_text(text)
    tfidf_features = tfidf.transform([cleaned_text])
    w2v_features = np.array([get_avg_word2vec(tokens, w2v_model.wv, 100)])
    combined_features = np.hstack((tfidf_features.toarray(), w2v_features))
    return combined_features, tfidf_features, tokens

# Function to get the average word2vec for a sentence
def get_avg_word2vec(tokens, model, vector_size):
    if len(tokens) < 1:
        return np.zeros(vector_size)
    vec = np.zeros(vector_size)
    count = 0
    for token in tokens:
        if token in model:
            vec += model[token]
            count += 1
    if count != 0:
        vec /= count
    return vec

# Function to make predictions on new data
def predict_and_explain(text):
    if '#disaster' in text.lower():
        return [1]  # Directly classify as disaster-related if hashtag is present
    features, tfidf_features, tokens = get_features(text)
    prediction = best_rf_model.predict(features)
    return prediction

# Function to extract tweet ID from URL
def extract_tweet_id(url):
    match = re.search(r'x.com/.*/status/(\d+)', url)
    if match:
        return match.group(1)
    else:
        raise ValueError("Invalid Twitter URL")

# Function to get tweet text using Twitter API
def get_tweet_text(tweet_id, bearer_token):
    url = f"https://api.twitter.com/2/tweets/{tweet_id}"
    headers = {
        "Authorization": f"Bearer {bearer_token}"
    }
    params = {
        "tweet.fields": "text"  # Request the text field of the tweet
    }
    response = requests.get(url, headers=headers, params=params)
    if response.status_code == 200:
        tweet_data = response.json()
        return tweet_data['data']['text']
    else:
        raise Exception(f"Request failed: {response.status_code}, {response.text}")

# Function to fetch recent disaster tweets
def fetch_recent_tweets(bearer_token, since_hours=None):
    url = "https://api.twitter.com/2/tweets/search/recent"
    headers = {
        "Authorization": f"Bearer {bearer_token}"
    }
    if since_hours is not None:
        start_time = (datetime.utcnow() - timedelta(hours=since_hours)).isoformat() + "Z"
    else:
        start_time = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0).isoformat() + "Z"
    
    params = {
        "query": "#Disaster",
        "max_results": 10,  # Adjust as needed
        "start_time": start_time,
        "tweet.fields": "created_at,text,geo",  # Include additional fields
        "expansions": "geo.place_id",  # Include place information
        "place.fields": "full_name"  # Include place name
    }
    response = requests.get(url, headers=headers, params=params)
    if response.status_code == 200:
        tweets_data = response.json()
        tweets = []
        for tweet in tweets_data['data']:
            tweet_info = {
                'text': tweet['text'],
                'created_at': tweet['created_at'],
                'place': next((place['full_name'] for place in tweets_data.get('includes', {}).get('places', []) if place['id'] == tweet.get('geo', {}).get('place_id')), 'Unknown')
            }
            tweet_info['prediction'] = predict_and_explain(tweet_info['text'])[0]
            tweets.append(tweet_info)
        return tweets
    else:
        raise Exception(f"Request failed: {response.status_code}, {response.text}")

# Custom CSS for background image
background_image_url = "https://images.pexels.com/photos/159490/yale-university-landscape-universities-schools-159490.jpeg?auto=compress&cs=tinysrgb&w=600"
background_style = f"""
    <style>
    .stApp {{
        background: url({background_image_url});
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
    }}
    </style>
    """

# Inject custom CSS
st.markdown(background_style, unsafe_allow_html=True)

# Streamlit UI
st.markdown('<h2 style="color: darkblue; background-color: rgba(255, 255, 255, 0.4); backdrop-filter: blur(5px); text-align: center; border-radius: 8px;">Disaster Tweet Classifier</h2>', unsafe_allow_html=True)
bearer_token = "AAAAAAAAAAAAAAAAAAAAAO9puQEAAAAAibBDDQEa%2BOYP%2BI%2BkQNdGzbfgOQQ%3DSryMUwRRPymuUcj0pA44K7TkHdRKXPE8LQrBeXkrQsIKgUHWDd"

# Input for tweet text or URL and classification
st.markdown('<h6 style="background-color: rgba(255, 255, 255, 0.4); backdrop-filter: blur(5px); margin-top: 20px; padding-top: 8px; border-radius: 8px; padding-left: 10px;">Classify Your Tweet:</h6>', unsafe_allow_html=True)
tweet_input = st.text_area('')
if st.button('Fetch & classify'):
    if tweet_input.strip():
        try:
            if tweet_input.startswith("http"):
                tweet_url = tweet_input.strip()
                tweet_id = extract_tweet_id(tweet_url)
                tweet_text = get_tweet_text(tweet_id, bearer_token)
                st.write(f"**Tweet Text:** {tweet_text}")
            else:
                tweet_text = tweet_input.strip()

            prediction = predict_and_explain(tweet_text)
            if prediction[0] == 1:
                st.markdown('<p style="color:red; background-color: rgba(255, 255, 255, 0.4); backdrop-filter: blur(5px); border-radius: 8px; padding: 10px;">This tweet is related to Disaster.</p>', unsafe_allow_html=True)
            else:
                st.markdown('<p style="color:green; background-color: rgba(255, 255, 255, 0.4); backdrop-filter: blur(5px); border-radius: 8px; padding: 10px;">This tweet is NOT related to Disaster.</p>', unsafe_allow_html=True)
        except Exception as e:
            st.error(f"Error: {e}")
    else:
        st.warning("Please enter a tweet text or URL.")

# Option to fetch recent disaster tweets
st.markdown('<h5 style="background-color: rgba(255, 255, 255, 0.4); backdrop-filter: blur(5px); border-radius: 8px; padding-left: 10px;">Fetch Recent Disaster Related Tweets</h5>', unsafe_allow_html=True)
fetch_option = st.selectbox("", ('Last 5 hours', 'Today'))

if st.button('Fetch Disaster Related Tweets'):
    if fetch_option == 'Last 5 hours':
        try:
            recent_tweets = fetch_recent_tweets(bearer_token, since_hours=5)
            if recent_tweets:
                st.write('### Recent Disaster Tweets (Last 5 hours):')
                for tweet in recent_tweets:
                    st.write(f"**Created At:** {tweet['created_at']}")
                    st.write(f"**Location:** {tweet['place']}")
                    st.write(f"**Tweet:** {tweet['text']}")
                    st.write(f"**Prediction:** {'Related to Disaster' if tweet['prediction'] == 1 else 'Not Related to Disaster'}")
                    st.write("---")
            else:
                st.info('No recent disaster tweets found.')
        except Exception as e:
            st.error(f"Error fetching recent tweets: {e}")

    elif fetch_option == 'Today':
        try:
            recent_tweets = fetch_recent_tweets(bearer_token)
            if recent_tweets:
                st.write('### Recent Disaster Tweets (Today):')
                for tweet in recent_tweets:
                    st.write(f"**Created At:** {tweet['created_at']}")
                    st.write(f"**Location:** {tweet['place']}")
                    st.write(f"**Tweet:** {tweet['text']}")
                    st.write(f"**Prediction:** {'Related to Disaster' if tweet['prediction'] == 1 else 'Not Related to Disaster'}")
                    st.write("---")
            else:
                st.info('No recent disaster tweets found.')
        except Exception as e:
            st.error(f"Error fetching recent tweets: {e}")
