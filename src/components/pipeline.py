from sklearn import pipeline
import pandas as pd
from src.components.preprocess import clean_text 
from src.components.sentiment import get_sentiment 
from src.components.summarization import generate_summary



try:
    SENTIMENT_CLASSIFIER = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
except Exception as e:
    print(f"Warning: Failed to load HuggingFace Sentiment pipeline: {e}")
    SENTIMENT_CLASSIFIER = None

def get_sentiment(text: str) -> dict:
    """
    Predicts sentiment and score using the pre-trained model.
    """
    if not SENTIMENT_CLASSIFIER or not text:
        return {"label": "N/A", "score": 0.0}
    
    result = SENTIMENT_CLASSIFIER(text, truncation=True)[0]
    
    return result


def run_pipeline_on_text(raw_text: str) -> dict:
    """Runs the full NLP pipeline on a single string of raw text."""
    
    if pd.isna(raw_text) or not raw_text:
        return {
            "raw_text_snippet": "N/A",
            "cleaned_text_snippet": "N/A",
            "predicted_sentiment": "N/A",
            "sentiment_score": "0.0000",
            "generated_summary": "N/A"
        }
        
    cleaned_text = clean_text(raw_text)
    sentiment_output = get_sentiment(raw_text)
    summary = generate_summary(raw_text)
    
    return {
        "raw_text_snippet": raw_text[:70] + "...",
        "cleaned_text_snippet": cleaned_text[:70] + "...",
        "predicted_sentiment": sentiment_output.get('label', 'ERROR'),
        "sentiment_score": f"{sentiment_output.get('score', 0.0):.4f}",
        "generated_summary": summary
    }

def run_pipeline_on_dataset(file_path: str, num_samples: int = 5):
    """Runs the pipeline on a subset of the Amazon dataset."""
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return

    sample_df = df.head(num_samples)
    results = []

    for _, row in sample_df.iterrows():
        raw_text = row['reviewText']
        
        if pd.isna(raw_text):
            continue
            
        pipeline_output = run_pipeline_on_text(raw_text)
        pipeline_output['overall_rating'] = row['overall']
        results.append(pipeline_output)
        
    results_df = pd.DataFrame(results)
    print("\n--- Pipeline Results on Amazon Dataset Samples ---")
    print(results_df.to_markdown(index=False, numalign="left", stralign="left"))
    
if __name__ == "__main__":
    run_pipeline_on_dataset("amazon.csv", num_samples=3)