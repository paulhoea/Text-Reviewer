# %%
import pandas as pd
import numpy as np
import nltk
import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import matplotlib.pyplot as plt
from wordcloud import WordCloud

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
fulltext = " ".join(df["review_text"].dropna())
fulltext = nltk.Text(nltk.word_tokenize(fulltext))

# %%
outlist = list()

for index, row in df.iterrows():
    tokens = nltk.wordpunct_tokenize(row["review_text"]) # instead of word_tokenize, to keep things like "Internet's" from splitting
    
    filtered_tokens = [word for word in tokens if word not in stop_words]
    filtered_tokens = [word for word in filtered_tokens if word not in punctuations]

    # ? Keep ? - splits the 's from things like "Internet's", test once I have more of a feel for it
    lemmatized_words = [lemmatizer.lemmatize(word) for word in filtered_tokens]

    outframe = pd.DataFrame({'word': lemmatized_words, 'rating': row["rating"], 'source': row["links"]})

    outlist.append(outframe)

text_frame = pd.concat(outlist)

# %%
fulltext.concordance("rattling")

# %%
x, y = np.ogrid[:300, :300]
mask = (x - 150) ** 2 + (y - 150) ** 2 > 130 ** 2
mask = 255 * mask.astype(int)

wc = WordCloud(background_color="white", repeat=True, mask=mask)
wc.generate(" ".join(text_frame["word"].dropna()))

plt.axis("off")
plt.imshow(wc, interpolation="bilinear")
plt.suptitle("All ratings", fontsize=14)
plt.show()

# %%

for key, group_df in text_frame.groupby("rating"):
    wc = WordCloud(background_color="white", repeat=True, mask=mask)
    wc.generate(" ".join(group_df["word"].dropna()))

    plt.axis("off")
    plt.imshow(wc, interpolation="bilinear")
    title = f"{group_df["rating"].unique()[0]}/10"
    plt.suptitle(title, fontsize=14)
    plt.show()




# %%
