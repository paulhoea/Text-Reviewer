# %%
import pandas as pd
import nltk
import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
nltk.data.path.append("/home/paul/Documents/Python Share/nltk_data")

# read scraped data
df = pd.read_csv("/home/paul/Documents/Text Reviewer/scraped_results.csv")
df["rating"] = df["rating"].str.split("/").str[0]
df["rating"] = pd.to_numeric(df["rating"]).astype('Int64')

# define stopwords and punctutation to remove
# stop_words = set(stopwords.words('english'))
punctuations = list(string.punctuation)
punctuations.append("''")
# lemmatizer = WordNetLemmatizer()


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

tokenized_reviews = list()

def get_wordnet_pos(tag):
    if tag.startswith('J'):
        return 'a'
    elif tag.startswith('V'):
        return 'v'
    elif tag.startswith('N'):
        return 'n'
    elif tag.startswith('R'):
        return 'r'
    else:
        return 'n'

for index, row in df.iterrows():
    print(f"Processing row #{index}")
    tokens = nltk.wordpunct_tokenize(row["review_text"].lower()) # instead of word_tokenize, to keep things like "Internet's" from splitting
    
    # Build a set of lowercase words to remove from artist and album names
    artist_words = set(nltk.wordpunct_tokenize(str(row["artist"]).lower()))
    album_words = set(nltk.wordpunct_tokenize(str(row["album"]).lower()))
    name_words = artist_words | album_words

    #filtered_tokens = [word for word in tokens if word not in stop_words]
    filtered_tokens = [word for word in tokens if word not in punctuations]

    #filtered_tokens = [word for word in filtered_tokens if word not in punctuations]
    filtered_tokens = [word for word in filtered_tokens if word not in name_words]  # remove artist/album words

    tagged_tokens = nltk.pos_tag(filtered_tokens)

    # ? Keep ? - splits the 's from things like "Internet's", test once I have more of a feel for it
    # lemmatized_words = pd.DataFrame(
    #     [(lemmatizer.lemmatize(word, get_wordnet_pos(tag)), tag) for word, tag in tagged_tokens],
    #     columns=["word", "pos"]
    #     )

    # lemmatized_words["rating"] = row["rating"]
    # lemmatized_words["source"] = row["links"]

    #tmp_review_df = pd.DataFrame({'word': lemmatized_words["word"], 'pos': lemmatized_words["pos"], 'rating': row["rating"], 'source': row["links"]})

    # tokenized_reviews.append(lemmatized_words)

    tagged_tokens = pd.DataFrame(tagged_tokens, columns=["word", "pos"])
    tokenized_reviews.append(tagged_tokens)

tokens_df = pd.concat(tokenized_reviews)

# %%
tokens_df.to_csv("/home/paul/Documents/Text Reviewer/tokens_df.csv", index=False)
with open("/home/paul/Documents/Text Reviewer/fulltext.txt", "w") as text_file:
    text_file.write(fulltext)
# %%
