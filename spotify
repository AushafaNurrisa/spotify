import streamlit as st
import pandas as pd

# Load the dataset
spotify_df = pd.read_csv("spotify.csv")

# Clean missing values
spotify_df = spotify_df[spotify_df['Title'].notnull()]

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

import re

# Define cleaning functions
clean_spcl = re.compile('[/(){}\[\]\|@,;]')
clean_symbol = re.compile('[^0-9a-z #+_]')
sastrawi = StopWordRemoverFactory()
stopworda = sastrawi.get_stop_words()
factory = StemmerFactory()
Stemmer = factory.create_stemmer()

def clean_text(text):
    text = text.lower()
    text = clean_spcl.sub(' ', text)
    text = clean_symbol.sub(' ', text)
    text = ' '.join(word for word in text.split() if word not in stopworda)
    return text

# Apply text cleaning
spotify_df['desc_clean'] = spotify_df['Title'].apply(clean_text)

# Set the index to 'Title'
spotify_df.set_index('Title', inplace=True)

# Generate TF-IDF matrix
tf = TfidfVectorizer(analyzer='word', ngram_range=(1, 3), min_df=0.0)
tfidf_matrix = tf.fit_transform(spotify_df['desc_clean'])
cos_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Create a Series to map indices
indices = pd.Series(spotify_df.index)

def recommendations(title, top=10):
    recommended_songs = []
    matching_indices = indices[indices.str.contains(title, case=False, na=False)]
    
    if matching_indices.empty:
        return ["No matches found for the input title."]
    
    idx = matching_indices.index[0]
    score_series = pd.Series(cos_sim[idx]).sort_values(ascending=False)
    
    top = top + 1
    top_indexes = list(score_series.iloc[:top].index)

    for i in top_indexes:
        recommended_songs.append(list(spotify_df.index)[i] + " - Popularity: " + str(spotify_df.iloc[i]['Popularity']))

    return recommended_songs

# Streamlit app
st.title("Sistem Rekomendasi Lagu")
lagu = st.text_input("Masukkan Judul Lagu")
rekomendasi = st.button("Rekomendasi")

if rekomendasi:
    hasil_rekomendasi = recommendations(lagu, 15)
    st.dataframe(hasil_rekomendasi)
