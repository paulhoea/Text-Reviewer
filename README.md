# Text Reviewer

This repository implements various techniques of natrual language processing (NLP) via Python's NLTK library. The program consists of three parts:
- A web scraper that extracts transcripts and ratings from [https://theneedledrop.com/](https://theneedledrop.com/) (a music review site)
- Processing of the data into tokenized and lemmatized dataframes for analysis
- A visualisation and analysis procedure that aims to extract useful information from the semi-structured text data

<img width="1134" height="989" alt="Term Counts" src="https://github.com/user-attachments/assets/a9f22e23-4139-4d4a-b8c3-ac31b2326377" />
<img width="1093" height="590" alt="TF-IDF Matrix" src="https://github.com/user-attachments/assets/aec944d3-cd6b-4e57-b494-9b2af5d5e852" />

## The following theories are used to guide the analysis:
- There might be a systematic difference between vocabularies used depending on numerical ratings (positive vs. negative phrasing)
- Discussion of instruments and terms related to production might cluster on very positive or negative reviews (since these aspects might be discussed in greater detail, while mediocre albums are described more generically)
- Certain descriptives might co-occur at higher rates (certain adjectives are strongly tied to particular nouns, such as "rumbling" bass or "lovely" singing) 

## Examples
Concordance wordclouds:

<img width="515" height="350" alt="Concordance_Vis" src="https://github.com/user-attachments/assets/bcc04691-a0cc-48b9-b613-d56996340cc3" />

... and multi-sting concordance:

<img width="927" height="345" alt="Concordance" src="https://github.com/user-attachments/assets/9f40e488-cd8e-452f-9b9f-c85af6d63079" />

The vocabulary of positive vs. negative reviews can be explored with rating group wordclouds:

<img width="389" height="437" alt="wordcloud" src="https://github.com/user-attachments/assets/d583a9c1-29a6-4e1a-8714-6d8865d89131" />
<img width="389" height="437" alt="wordcloud2" src="https://github.com/user-attachments/assets/e66a3c72-21b9-490f-ac78-e34bc073b375" />

