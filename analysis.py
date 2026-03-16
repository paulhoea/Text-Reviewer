# %% TODO: explore this further
# fulltext.concordance("mix") # note: full text is not lemmatized
# fulltext.concordance("mixing")

# search for variations of "mixing"
wildcard_pattern = r'\bmix(ing|es|er|s)?\b'
width = 5  # tokens of context either side

# Find indices of matching tokens
match_indices = [i for i, word in enumerate(fulltext.tokens) 
                 if re.search(wildcard_pattern, word, re.IGNORECASE)]

# Print concordance-style output
print(f"{'LEFT CONTEXT':>50}  {'MATCH':<10}  {'RIGHT CONTEXT'}")
print("-" * 80)
for i in match_indices:
    left = fulltext.tokens[max(0, i-width):i]
    match = fulltext.tokens[i]
    right = fulltext.tokens[i+1:i+width+1]
    
    left_str = ' '.join(left).rjust(50)
    right_str = ' '.join(right)
    print(f"{left_str}  {match:<10}  {right_str}")

# %%
# search for co-occurance around a specific word
fulltext.concordance("rattling")
search_word = "rattling"

concordance_results = fulltext.concordance_list(search_word)
concordance_tokens = []

for result in concordance_results:
    tokens = nltk.wordpunct_tokenize(result.line)

    # remove stopwords
    filtered_tokens = [word for word in tokens if word not in stop_words]
    filtered_tokens = [word for word in filtered_tokens if word not in punctuations]
    filtered_tokens = [word for word in filtered_tokens if word not in name_words]

    # remove original search word
    filtered_tokens.remove(search_word)

    concordance_tokens.append(pd.DataFrame({"word": filtered_tokens, "search_word": search_word}))

concordance_df = pd.concat(concordance_tokens)


# %% find instruments in the text
def get_instrument_synsets():
    """Get all hyponyms of 'musical instrument' from WordNet."""
    instrument_synset = wn.synset("musical_instrument.n.01")
    hyponyms = instrument_synset.closure(lambda s: s.hyponyms())
    return {lemma.name().lower().replace("_", " ")
            for synset in hyponyms
            for lemma in synset.lemmas()}

instruments = get_instrument_synsets()

# Visualise top-used instruments
fulltext.dispersion_plot(list(instruments))
# TODO: Define a custom list of music terms (instruments, production terms) and plot their frequencies. Grouping by rating with a grouped or stacked bar chart adds an extra layer.
# TODO: TF-IDF Heatmap rating X music terms x freqency (Compute TF-IDF scores across reviews with sklearn's TfidfVectorizer, then plot a heatmap where rows are output_group bins and columns are your top music terms. This shows which terms are distinctively used in high vs. low-rated reviews, not just frequent ones.)
# TODO: Plot mean rating by term: Violin/Box Plot of Scores by Term Presence OR Calculate the mean rating for reviews containing each term, plot as a dot with confidence intervals.

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
width, height = 1200, 600

img = Image.new("L", (width, height), 255)  # white background
draw = ImageDraw.Draw(img)

# choose a font (adjust path as needed)
font_path = font_manager.findfont("Bold")
font = ImageFont.truetype(font_path, 400)

# set mask to the shape of the input word(s)
search_words = concordance_df['search_word'].unique()

if len(search_words) == 1:
    mask_word = search_words[0].upper()
else:
    mask_word = ", ".join(search_words).upper()

# center text
bbox = draw.textbbox((0, 0), mask_word, font=font)
text_w = bbox[2] - bbox[0]
text_h = bbox[3] - bbox[1]

x = (width - text_w) // 2
y = (height - text_h) // 2

# draw text in black
draw.text((x, y), mask_word, fill=0, font=font)

text_mask = np.array(img)

# generate wordcloud for concordance_df
wc = WordCloud(background_color="white", repeat=True, mask=text_mask)
wc.generate(" ".join(concordance_df["word"].dropna()))

plt.axis("off")
plt.imshow(wc, interpolation="bilinear")
title = mask_word
plt.suptitle(title, fontsize=14)
plt.show()
# %%