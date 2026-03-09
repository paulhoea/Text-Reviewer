# %%
import pandas as pd
import numpy as np
import nltk
import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from nltk.corpus import wordnet as wn

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

# %% TODO: explore this further, after identifying instruments for example
fulltext.concordance("rattling")
fulltext.concordance("mix")
fulltext.concordance("mixing")


# %% find instruments in the text
def get_instrument_synsets():
    """Get all hyponyms of 'musical instrument' from WordNet."""
    instrument_synset = wn.synset("musical_instrument.n.01")
    hyponyms = instrument_synset.closure(lambda s: s.hyponyms())
    return {lemma.name().lower().replace("_", " ")
            for synset in hyponyms
            for lemma in synset.lemmas()}

instruments = get_instrument_synsets()


# %%
# circle shape for the wordcloud
x, y = np.ogrid[:300, :300]
mask = (x - 150) ** 2 + (y - 150) ** 2 > 130 ** 2
mask = 255 * mask.astype(int)

# generate wordcloud
wc = WordCloud(background_color="white", repeat=True, mask=mask)
wc.generate(" ".join(tokens_df["word"].dropna()))

plt.axis("off")
plt.imshow(wc, interpolation="bilinear")
plt.suptitle("All ratings", fontsize=14)
plt.show()

# %%

tokens_df["output_group"] = tokens_df["rating"].where(tokens_df["rating"] < 9, "9+")

# %% 
# Filtering out the words that are in the top fifty of more than five groups after main wordcloud, to get more differentiated results

top_words_by_group = []

for key, group_df in tokens_df.groupby("output_group"):
    top_words_in_group = group_df[["word", "output_group"]].value_counts().head(50).reset_index()
    top_words_by_group.append(top_words_in_group)

top_words = pd.concat(top_words_by_group)
top_words = top_words["word"].value_counts().reset_index()
top_words = top_words[top_words["count"] > 5]

tokens_df = tokens_df[~tokens_df["word"].isin(top_words["word"])]


# %% 
# generate wordcloud for each score category from 0/10 to 10/10
for key, group_df in tokens_df.groupby("output_group"):
    wc = WordCloud(background_color="white", repeat=True, mask=mask)
    wc.generate(" ".join(group_df["word"].dropna()))

    plt.axis("off")
    plt.imshow(wc, interpolation="bilinear")
    title = f"{group_df["output_group"].unique()[0]}/10"
    plt.suptitle(title, fontsize=14)
    plt.show()




# %%
