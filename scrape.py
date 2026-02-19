# %% SETUP
import requests
import pandas as pd
import xmltodict

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

# %% filter only album reviews
inputlinks["relevant"] = inputlinks["links"].str.contains("album-reviews")
inputlinks = inputlinks[inputlinks["relevant"] == True]

# filter to date where transcripts are availible
# NOT NESSECARY AS THE NAMING FORMAT WITH "album-review" in the URL was only adopted here, but kept for completeness
inputlinks = inputlinks[inputlinks["lastmod"] > "2024-07-31"] 

# reset index for further processing
inputlinks.reset_index(drop = True, inplace = True)
# %%
