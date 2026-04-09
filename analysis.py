# %%
import pandas as pd
import numpy as np
import nltk
import string
from nltk.corpus import stopwords
from nltk.probability import FreqDist
import matplotlib.pyplot as plt
import cv2
from matplotlib import font_manager
from wordcloud import WordCloud
from nltk.corpus import wordnet as wn
from PIL import Image, ImageDraw, ImageFont
from pyfonts import load_google_font
from textwrap import wrap
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
import seaborn as sns
import matplotlib.gridspec as gridspec
# if nessecary, path to nltk resources
nltk.data.path.append("/home/paul/Documents/Python Share/nltk_data")

# %% import processed data
with open("/home/paul/Documents/Text Reviewer/fulltext.txt", "r", encoding="utf-8") as file:
        fulltext = file.read()
fulltext = nltk.Text(nltk.word_tokenize(fulltext))
tokens_df = pd.read_csv("/home/paul/Documents/Text Reviewer/tokens_df.csv")

# define stopwords and punctutation to remove
stop_words = set(stopwords.words('english'))
punctuations = list(string.punctuation)
punctuations.append("''")






# %%
# Concordance: search for variations of "mixing"´
# input_words = ["mixing", "the mix", "a mix of"]
input_words = ["rattling"]
width = 7  # tokens of context either side

# split the input words into words or phrases, as these need separate scanning
single_words = set(w.lower() for w in input_words if len(w.split()) == 1)
phrases = [p.lower().split() for p in input_words if len(p.split()) > 1]

# extract tokens from the larger DF for performance reasons
tokens_list = tokens_df.word.tolist()

# Single word matches, returns the index of the match
match_indices = [i for i, word in enumerate(tokens_list)
                 if word in single_words]

# Phrase matches, with shifted indices so that the multi-word phrase works in the display
for phrase in phrases:
    phrase_len = len(phrase)
    for i in range(len(tokens_list) - phrase_len + 1):
        if list(tokens_list[i:i+phrase_len]) == phrase:
            match_indices.append(i)

match_indices = sorted(set(match_indices))

# create results dataframe
results_list = []

for idx, i in enumerate(match_indices):
    match_len = next(
        (len(p) for p in phrases if tokens_list[i:i+len(p)] == p), 1
    )

    # filter out context from surrounding tokens; total line length is width * 2 + match_length
    left = tokens_list[max(0, i-width):i] # list
    match = ' '.join(tokens_list[i:i+match_len]) # string
    right = tokens_list[i+match_len:i+match_len+width] # list

    # iterate over each word in left + right
    for word in left + right:
        results_list.append({
            "match_id": idx,
            "word": word,
            "search_word": match
        })

concordance_df = pd.DataFrame(results_list)

# Print concordance output
print(f"Concordance of {input_words}")
print(f"{len(match_indices)} total results")
print(f"{'LEFT CONTEXT':>50}  {'MATCH':<10}  {'RIGHT CONTEXT'}")
print("-" * 80)
# ----
for match_id, group in concordance_df.groupby("match_id"):
    words = group["word"].tolist()
    match = group["search_word"].iloc[0]

    # split into left/right using width; the first (width) words in the column are left, the ones after are right of the search term. Results in a total of (width) * 2 + match_len words being displayed
    left = words[:width]
    right = words[width:]

    left_str = ' '.join(left).rjust(50)
    right_str = ' '.join(right)

    print(f"{left_str}  {match:<10}  {right_str}")

    # create and print strings for the display
    # left_str = ' '.join(left).rjust(50) # turns filtered list from previous step to string
    # right_str = ' '.join(right) # turns filtered list from previous step to string
    # print(f"{left_str}  {match:<10}  {right_str}")









# %%
# Concordance: visualize one specific word as a wordcloud
# search_word = "rattling"

# concordance_results = fulltext.concordance_list(search_word)
# concordance_tokens = []

# for result in concordance_results:
#     tokens = nltk.wordpunct_tokenize(result.line)

#     # remove stopwords
#     filtered_tokens = [word for word in tokens if word not in stop_words]
#     filtered_tokens = [word for word in filtered_tokens if word not in punctuations]

#     # remove original search word
#     filtered_tokens.remove(search_word)

#     concordance_tokens.append(pd.DataFrame({"word": filtered_tokens, "search_word": search_word}))

# concordance_df = pd.concat(concordance_tokens)

# Prepare mask
width, height = 1200, 800
img_mask = Image.new("L", (width, height), 255)  # white background
draw = ImageDraw.Draw(img_mask)

# load Google font and convert to PIL font
font_prop_mask = load_google_font("Google Sans Code", weight="extra-bold")
font_path_mask = font_manager.findfont(font_prop_mask)
font_mask = ImageFont.truetype(font_path_mask, size=300)   # set desired size

# set mask to the shape of the input word(s)
search_words = concordance_df['search_word'].unique()
mask_word = "".join(search_words).upper()
mask_word = "\n".join(wrap(mask_word, 4))
# title_word = ", ".join(search_words).upper()

# text placement calulation
bbox = draw.textbbox((0, 0), mask_word, font=font_mask)
text_w = bbox[2] - bbox[0]
text_h = bbox[3] - bbox[1]
x = (width - text_w) // 2
y = (height - text_h) // 3

# draw text, convert to mask, and invert
draw.text((x, y), mask_word, fill=0, font=font_mask)
text_mask = np.array(img_mask)
inverted_mask = 255 - text_mask

# create outline image as an underlay
img_outline = Image.new("RGB", (width, height), color=(249, 242, 141))  # yellow background
draw = ImageDraw.Draw(img_outline)

# render outline image
font_prop_outline = load_google_font("Google Sans Code", weight="medium")
font_path_outline = font_manager.findfont(font_prop_outline)
font_outline = ImageFont.truetype(font_path_outline, size=300)   # set desired size
draw.text((x, y), mask_word, fill="white", font=font_outline)

# calculate colour mapping
frequencies = concordance_df["word"].dropna().value_counts().to_dict()
sorted_words = sorted(frequencies, key=frequencies.get, reverse=True)
rank = {w: i for i, w in enumerate(sorted_words)}
max_rank = max(rank.values())

c1 = np.array([59, 76, 192])    # blue
c2 = np.array([255, 204, 0])    # orange
c3 = np.array([237, 33, 0])     # red/purple

def interpolate(ca, cb, t):
    return (1 - t) * ca + t * cb

def color_func(word, *args, **kwargs):
    norm = rank[word] / max_rank if max_rank > 0 else 0

    if norm < 0.5:
        t = norm * 2
        color = interpolate(c1, c2, t)
    else:
        t = (norm - 0.5) * 2
        color = interpolate(c2, c3, t)

    r, g, b = color.astype(int)
    return f"rgb({r}, {g}, {b})"

# generate wordcloud
wc = WordCloud(
    background_color="rgba(255, 255, 255, 0)", 
    mode="RGBA", 
    repeat=True, 
    mask=inverted_mask, 
    color_func=color_func)
wc.generate_from_frequencies(frequencies)

plt.imshow(img_outline, cmap='gray', vmin=0, vmax=255)
plt.imshow(wc, interpolation="bilinear")
plt.axis("off")
plt.show()


# %%
# new search method, manual filtering with informed filters
tokens_df["prev_word"] = tokens_df.shift(1)["word"]
tokens_df["prev_pos"] = tokens_df.shift(1)["pos"]
tokens_df["next_word"] = tokens_df.shift(-1)["word"]
tokens_df["next_pos"] = tokens_df.shift(-1)["pos"]

tokens_df[tokens_df["word"].str.contains("mix")]

target_df = tokens_df[tokens_df["word"].str.contains("mix", na=False)].copy()

target_df["prev_pos_word"] = target_df["prev_pos"] + ", " + target_df["prev_word"] + " + " + target_df["pos"] +  ", " + target_df["word"]

prev_word_counts = (
    target_df["prev_word"]
    .dropna()
    .value_counts()
    .head(50)
    .sort_values()  # ascending so top rank is at the top of horizontal bar
)

prev_combo_counts = (
    target_df["prev_pos_word"]
    .dropna()
    .value_counts()
    .head(50)
    .sort_values()
)

fig = plt.figure(figsize=(14, 10))
fig.suptitle('Preceding Context of words containing "mix"', fontsize=15, fontweight="bold", y=1.01)
gs = gridspec.GridSpec(1, 2, figure=fig, wspace=0.6)

# Color palette
color_word = "#4C72B0"
color_combo = "#DD8452"

# -- Left: prev_word --
ax1 = fig.add_subplot(gs[0])
bars1 = ax1.barh(prev_word_counts.index, prev_word_counts.values, color=color_word, edgecolor="white", linewidth=0.5)
ax1.set_title("Preceding Word", fontsize=13, fontweight="bold", pad=10)
ax1.set_xlabel("Count", fontsize=11)
ax1.set_ylabel("prev_word", fontsize=11)
ax1.bar_label(bars1, padding=3, fontsize=9)
ax1.spines[["top", "right"]].set_visible(False)
ax1.tick_params(axis="y", labelsize=9)

# -- Right: prev_pos + prev_word --
ax2 = fig.add_subplot(gs[1])
bars2 = ax2.barh(prev_combo_counts.index, prev_combo_counts.values, color=color_combo, edgecolor="white", linewidth=0.5)
ax2.set_title("Preceding POS + Word", fontsize=13, fontweight="bold", pad=10)
ax2.set_xlabel("Count", fontsize=11)
ax2.set_ylabel("prev_pos + prev_word", fontsize=11)
ax2.bar_label(bars2, padding=3, fontsize=9)
ax2.spines[["top", "right"]].set_visible(False)
ax2.tick_params(axis="y", labelsize=9)

plt.tight_layout()
plt.show()







# manually? "song", "dense", "vocal", "dip", "vocal", "sound", "distortion", "sound", "master", "good", "bad", "clarity", "mastering", "track", "EQ", "eqed", "vocal", "track", "vocal" 37537

# %% find instruments in the text
def get_instrument_synsets():
    """Get all hyponyms of 'musical instrument' from WordNet."""
    instrument_synset = wn.synset("musical_instrument.n.01")
    hyponyms = instrument_synset.closure(lambda s: s.hyponyms())
    return {lemma.name().lower().replace("_", " ")
            for synset in hyponyms
            for lemma in synset.lemmas()}

instruments = get_instrument_synsets()

# manually add production terms
# source: https://routenote.com/blog/music-production-terms-a-glossary/
production_terms = {"mixing", "mastering", "mix", "master", "mixed", "mastered", "clipping", "compression", "compress", "equalisation", "EQ", "reverb"}

search_terms = instruments | production_terms

# Count only tokens that are instruments
# musical_word_counts = Counter(
#     word for word in tokens_df["word"] if word in search_terms
# )

# %%
musical_word_counts = (
    tokens_df[tokens_df["word"].isin(search_terms)]
    .groupby(["rating", "word"])
    .size()
    .reset_index(name="count")
)

pivot = musical_word_counts.pivot(index="word", columns="rating", values="count").fillna(0)

# Total tokens per rating group (the denominator)
total_words_by_rating = tokens_df.groupby("rating")["word"].count()

# Divide each count by the total tokens in that rating
pivot_normalized = pivot.div(total_words_by_rating, axis="columns")

# sorts by most common
pivot_normalized = pivot_normalized.loc[pivot_normalized.sum(axis=1).sort_values(ascending=False).index]

# selects top 20
pivot_normalized = pivot_normalized.iloc[0:20]

plt.figure(figsize=(12, 10))
sns.heatmap(pivot_normalized, cmap="YlOrRd", linewidths=0.5)
plt.title("Normalized Counts of Musical Terms")
plt.xlabel("Rating")
plt.ylabel("Term")
plt.tight_layout()
plt.show()

# Visualise top-used instruments
fulltext.dispersion_plot(list(instruments))
# TODO: Define a custom list of music terms (instruments, production terms) and plot their frequencies. Grouping by rating with a grouped or stacked bar chart adds an extra layer.
# TODO: TF-IDF Heatmap rating X music terms x freqency (Compute TF-IDF scores across reviews with sklearn's TfidfVectorizer, then plot a heatmap where rows are output_group bins and columns are your top music terms. This shows which terms are distinctively used in high vs. low-rated reviews, not just frequent ones.)
# - TODO: ad wordclouds: TF-IDF weighting; Instead of raw frequency, weight each word by how distinctive it is to that rating group relative to the whole corpus. Words like "track" and "sound" appear everywhere so their IDF score will be low, naturally suppressing them.
#         You'd compute TF-IDF with sklearn, then use the scores as the word weights passed to WordCloud(frequencies=...) instead of raw counts.
# TODO: Plot mean rating by term: Violin/Box Plot of Scores by Term Presence OR Calculate the mean rating for reviews containing each term, plot as a dot with confidence intervals.


filtered_df = tokens_df[tokens_df['word'].isin(production_terms)]

docs = (
    filtered_df
    .groupby('rating')['word']
    .apply(' '.join)
    .reset_index()
)

vectorizer = TfidfVectorizer(vocabulary=production_terms)
tfidf_matrix = vectorizer.fit_transform(docs['word'])
feature_names = vectorizer.get_feature_names_out()


viz_df = pd.DataFrame(
    tfidf_matrix.toarray(),
    index=docs['rating'],
    columns=feature_names
)


plt.figure(figsize=(12, 6))
sns.heatmap(
    viz_df,
    annot=False, fmt=".2f",
    cmap="YlOrRd",
    linewidths=0.5,
    cbar_kws={'label': 'TF-IDF Score'}
)
plt.title("TF-IDF Matrix — Selected Terms by Rating")
plt.xlabel("Term")
plt.ylabel("Rating")
plt.tight_layout()
plt.show()







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
    title = f"Rating: {group_df["output_group"].unique()[0]}/10"
    plt.suptitle(title, fontsize=14)
    plt.show()
