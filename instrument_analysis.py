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

