import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk
from src.components.preprocess import clean_text
from src.components.sentiment import create_sentiment_label

def prepare_data_for_training(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applies cleaning and labeling to the Amazon dataset.
    This function is primarily used by src/train.py.
    """
    df = df.dropna(subset=['reviewText'])
    
    df['cleaned_review'] = df['reviewText'].apply(clean_text)
    
    df['sentiment_label'] = df['overall'].apply(create_sentiment_label)
    
    return df[['cleaned_review', 'sentiment_label']]