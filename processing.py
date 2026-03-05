# %%
import pandas as pd
import nltk
import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


# read scraped data
df = pd.read_csv("/home/paul/scraped_results.csv")
# %%
fulltext = " ".join(df["review_text"].dropna())
tokens = nltk.word_tokenize(fulltext.lower())
review_text = nltk.Text(tokens)

stop_words = set(stopwords.words('english'))
punctuations = list(string.punctuation)
punctuations.append("''")

filtered_tokens = [word for word in tokens if word not in stop_words]
filtered_tokens = [word for word in filtered_tokens if word not in punctuations]

# ??
lemmatizer = WordNetLemmatizer()
lemmatized_words = [lemmatizer.lemmatize(word) for word in filtered_tokens]

# TODO: come up with a data structure
df[["links", "rating", "review_text"]]

# %%
review_text.concordance("rattling")

# %%







