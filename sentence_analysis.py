# %%
import pandas as pd
import numpy as np
import nltk
import string
from nltk.corpus import stopwords
import matplotlib.pyplot as plt
from matplotlib import font_manager
from wordcloud import WordCloud
from PIL import Image, ImageDraw, ImageFont
from pyfonts import load_google_font
from textwrap import wrap

# if nessecary, path to nltk resources
nltk.data.path.append("/home/paul/Documents/Python Share/nltk_data")

# %% import processed data
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

concordance_df = concordance_df[
    ~concordance_df['word'].isin(stop_words) & 
    ~concordance_df['search_word'].isin(stop_words)
]

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