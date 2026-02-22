# %% SETUP
import requests
import pandas as pd
import xmltodict
from bs4 import BeautifulSoup
import re
import time

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


### SCRAPING START ###

# %% Test request and HTML parsing
# inputlinks = inputlinks[inputlinks.index == 10]
output = inputlinks.copy() # copies the links list to be extended with relevant information

for index, row in inputlinks.iterrows():
    response = requests.get(row["links"], timeout=2)

    time.sleep(0.5)

    output.loc[index, "status_code"] = response.status_code
    print(f"Working on link {index} out of {len(inputlinks)}, response code {response.status_code}")

    if response: # Note: "response automatically converts to boolean TRUE if successful (<400), and FALSE for unsuccessful"
        output.loc[index, "raw_html"] = response.text
        parsed_html = BeautifulSoup(response.text, "html.parser")

        # <meta content="9/10" property="article:tag"/>; question if this is universal. One approach would be to extract these via regex
        rating = None
        for tag in parsed_html.find_all("meta", property="article:tag"):
            content = tag.get("content", "")
            if re.fullmatch(r"\d+/10", content):
                rating = content
                break
        output.loc[index, "rating"] = rating

        review_text = parsed_html.find("div", {"class":"post_content"}).text
        output.loc[index, "review_text"] = review_text
    else:
        print(response.headers)
        raise Exception(f"Non-success status code: {response.status_code}")

output
#print(output.review_text.iloc[0])

output.to_csv("/home/paul/scraped_results.csv", index=False)

# %%
