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
