from src.components.preprocess import clean_text


def create_sentiment_label(rating: int) -> str:
    """Converts the 1-5 star rating into a sentiment label."""
    if rating in [1, 2]:
        return "Negative"
    elif rating == 3:
        return "Neutral"
    elif rating in [4, 5]:
        return "Positive"
    else:
        return "Unknown"