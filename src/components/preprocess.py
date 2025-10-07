import re
import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk


def clean_text(text: str) -> str:
    """Performs light cleaning for transformer models (only handles strings)."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'[\n\t]', ' ', text)  
    text = re.sub(r'[^a-z0-9\s.,?!]', '', text) 
    
    word_tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    filtered_words = [w for w in word_tokens if w not in stop_words]
    
    return ' '.join(filtered_words)