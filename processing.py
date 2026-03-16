# %%
import pandas as pd
import numpy as np
import nltk
import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import matplotlib.pyplot as plt
from matplotlib import font_manager
from wordcloud import WordCloud
from nltk.corpus import wordnet as wn
from PIL import Image, ImageDraw, ImageFont
import re

# read scraped data
df = pd.read_csv("/home/paul/scraped_results.csv")
df["rating"] = df["rating"].str.split("/").str[0]
df["rating"] = pd.to_numeric(df["rating"]).astype('Int64')

# define stopwords and punctutation to remove
stop_words = set(stopwords.words('english'))
punctuations = list(string.punctuation)
punctuations.append("''")
lemmatizer = WordNetLemmatizer()


# %% 
# generate a column which contains scores processed from text if no numerical rating could be scraped

number_words = {
    'zero': '0', 'one': '1', 'two': '2', 'three': '3', 'four': '4', 'five': '5',
    'six': '6', 'seven': '7', 'eight': '8', 'nine': '9', 'ten': '10'
}

df['rating_from_scrape'] = df['rating']

df['rating'] = df['rating_from_scrape'].fillna(
    df['rating_text']
    .str.lower()
    .replace(number_words, regex=True)
    .str.extractall(r'(\d+(?:\.\d+)?)') # returns one row per match, so ranges like "four to five" produce two rows with a multi-level index (original_index, match_number)
    [0] # selects only the first of this index
    .astype(float)
    .groupby(level=0) # group by original_index to collapse multiple matches per row back into one value
    .mean()
    .round()
)

# the remaining NAs are assumed to be non-standard ratings, and thus dropped
df.dropna(subset=["rating"], inplace=True)


# %%
fulltext = " ".join(df["review_text"].dropna())
fulltext = nltk.Text(nltk.word_tokenize(fulltext))

# %%
tokenized_reviews = list()

for index, row in df.iterrows():
    tokens = nltk.wordpunct_tokenize(row["review_text"].lower()) # instead of word_tokenize, to keep things like "Internet's" from splitting
    
    # Build a set of lowercase words to remove from artist and album names
    artist_words = set(nltk.wordpunct_tokenize(str(row["artist"]).lower()))
    album_words = set(nltk.wordpunct_tokenize(str(row["album"]).lower()))
    name_words = artist_words | album_words

    filtered_tokens = [word for word in tokens if word not in stop_words]
    filtered_tokens = [word for word in filtered_tokens if word not in punctuations]
    filtered_tokens = [word for word in filtered_tokens if word not in name_words]  # remove artist/album words

    # ? Keep ? - splits the 's from things like "Internet's", test once I have more of a feel for it
    lemmatized_words = [lemmatizer.lemmatize(word) for word in filtered_tokens]

    tmp_review_df = pd.DataFrame({'word': lemmatized_words, 'rating': row["rating"], 'source': row["links"]})

    tokenized_reviews.append(tmp_review_df)

tokens_df = pd.concat(tokenized_reviews)

