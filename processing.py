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
df.dropna(subset=["rating"], inplace=True)

# define stopwords and punctutation to remove
stop_words = set(stopwords.words('english'))
punctuations = list(string.punctuation)
punctuations.append("''")
lemmatizer = WordNetLemmatizer()


# %%
fulltext = " ".join(df["review_text"].dropna())
fulltext = nltk.Text(nltk.word_tokenize(fulltext))

# %%
tokenized_reviews = list()

for index, row in df.iterrows():
    tokens = nltk.wordpunct_tokenize(row["review_text"].lower()) # instead of word_tokenize, to keep things like "Internet's" from splitting
    
    filtered_tokens = [word for word in tokens if word not in stop_words]
    filtered_tokens = [word for word in filtered_tokens if word not in punctuations]

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


# %% TODO: apply new filtering

tokens_df['rating'].unique()

def find_common_top_words(tokens_df, top_n=10):
    """
    Find words that appear in the top `top_n` most frequent words across groups
    
    Returns:
        DataFrame with words and the number of groups they appear in
    """
    groups = tokens_df['rating'].unique()
    
    # For each group, get the top N words by frequency
    top_words_per_group = {}
    for group in groups:
        group_df = tokens_df[tokens_df['rating'] == group]
        top_words = (
            group_df['word']
            .value_counts()
            .head(top_n)
            .index
            .tolist()
        )
        top_words_per_group[group] = set(top_words)
    
    # Count how many groups each word appears in
    all_top_words = [word for words in top_words_per_group.values() for word in words]
    word_group_counts = pd.Series(all_top_words).value_counts()
    
    return word_group_counts

find_common_top_words(tokens_df, top_n=10)












# %% TODO: REPLACE
# Filtering out the top ten words after main wordcloud, to get more differentiated results

top_words = tokens_df["word"].value_counts().head(10).index
tokens_df = tokens_df[~tokens_df["word"].isin(top_words)]

# %%

# generate wordcloud for each score category from 0/10 to 10/10
for key, group_df in tokens_df.groupby("rating"):
    wc = WordCloud(background_color="white", repeat=True, mask=mask)
    wc.generate(" ".join(group_df["word"].dropna()))

    plt.axis("off")
    plt.imshow(wc, interpolation="bilinear")
    title = f"{group_df["rating"].unique()[0]}/10"
    plt.suptitle(title, fontsize=14)
    plt.show()




# %%
