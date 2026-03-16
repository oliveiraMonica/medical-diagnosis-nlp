"""
Text preprocessing module for the Medical NLP project.

This module contains functions responsible for cleaning and preparing
clinical text before it is used by machine learning models.

Typical preprocessing steps include:
- Lowercasing text
- Removing punctuation
- Removing stopwords
- Basic normalization

These steps help transform raw clinical notes into structured text
that can be used for NLP feature extraction.
"""

import re
import nltk
from nltk.corpus import stopwords

stop_words = set(stopwords.words("english"))

def clean_text(text):
    """
    Clean clinical transcription text for NLP processing.
    Steps:
    - convert to lowercase
    - remove non-letter characters
    - remove stopwords
    """

    # convert to lowercase
    text = text.lower()

    # remove numbers and punctuation
    text = re.sub(r"[^a-zA-Z\s]", "", text)

    # tokenize
    tokens = text.split()

    # remove stopwords
    tokens = [word for word in tokens if word not in stop_words]

    # join tokens again
    cleaned_text = " ".join(tokens)

    return cleaned_text