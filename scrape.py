# %% SETUP
import requests
import pandas as pd
import xmltodict
from bs4 import BeautifulSoup
import re

# paths
# path_input = "/home/paul/Documents/TextReviewer/Input/sitemap-posts.xml"

# %%
# get sitemap
url = "https://theneedledrop.com/sitemap-posts.xml"
res = requests.get(url)
raw = xmltodict.parse(res.text)

# process XML to dataframe
inputlinks = [[r["loc"], r["lastmod"]] for r in raw["urlset"]["url"]]
print("Number of URLs:", len(inputlinks))
inputlinks = pd.DataFrame(inputlinks, columns=["links", "lastmod"])
inputlinks["lastmod"] = pd.to_datetime(inputlinks["lastmod"])

# %% filter only album reviews, and remove staff reviews
inputlinks = inputlinks[inputlinks["links"].str.contains("album-reviews")]
inputlinks = inputlinks[~inputlinks["links"].str.contains("tnd-staff")]

# filter to date where transcripts are availible
# NOT NESSECARY AS THE NAMING FORMAT WITH "album-review" in the URL was only adopted here, but kept for completeness
inputlinks = inputlinks[inputlinks["lastmod"] > "2024-07-31"] 

# reset index for further processing
inputlinks.reset_index(drop = True, inplace = True)


# %% Test request and HTML parsing
x = inputlinks[inputlinks.index == 10]

for index, row in x.iterrows():
    print(row["links"])
    page = requests.get(row["links"])

    x.loc[index, "raw_html"] = page.text

    parsed_html = BeautifulSoup(page.text, "html.parser")

    # <meta content="9/10" property="article:tag"/>; question if this is universal. One approach would be to extract these via regex
    rating = None
    for tag in parsed_html.find_all("meta", property="article:tag"):
        content = tag.get("content", "")
        if re.fullmatch(r"\d+/10", content):
            rating = content
            break
    x.loc[index, "rating"] = rating

    review_text = parsed_html.find("div", {"class":"post_content"}).text
    print(review_text)
    x.loc[index, "review_text"] = review_text
    
x
print(x.review_text.iloc[0])


# %%
