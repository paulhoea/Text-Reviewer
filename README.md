# Text Reviewer

This repository implements various techniques of natrual language processing (NLP) via Python's NLTK library. The program consists of three parts:
- A web scraper that extracts transcripts and ratings from [https://theneedledrop.com/](https://theneedledrop.com/) (a music review site)
- Processing of the data into tokenized and lemmatized dataframes for analysis
- A visualisation and analysis procedure that aims to extract useful information from the semi-structured text data

The following theories are used to guide the analysis:
- There might be a systematic difference between vocabularies used depending on numerical ratings (positive vs. negative phrasing)
- Discussion of instruments and terms related to production might cluster on very positive or negative reviews (since these aspects might be discussed in greater detail, while mediocre albums are described more generically)
- Certain descriptives might co-occur at higher rates (certain adjectives are strongly tied to particular nouns, such as "rumbling" bass or "lovely" singing) 
