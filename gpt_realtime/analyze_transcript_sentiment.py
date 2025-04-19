import os
import json
import glob
from datetime import datetime
import openai

def load_api_keys():
    """Load API keys from .api_key.json file."""
    try:
        # Look for the key file in the parent directory relative to this script
        script_dir = os.path.dirname(__file__)
        potential_path = os.path.join(script_dir, '..', '.api_key.json')
        if os.path.exists(potential_path):
            api_key_path = potential_path
        elif os.path.exists('.api_key.json'):
            api_key_path = '.api_key.json'
        else:
            raise FileNotFoundError("API key file not found")

        with open(api_key_path, "r") as f:
            api_keys = json.load(f)
            return api_keys.get("openai_api_key")
    except Exception as e:
        print(f"Error loading API keys: {e}")
        return None

def get_latest_files(transcripts_dir):
    """Get the latest conversation transcript and sentiment results files."""
    # Get all transcript files
    transcript_files = glob.glob(os.path.join(transcripts_dir, 'conversation_*.txt'))
    sentiment_files = glob.glob(os.path.join(transcripts_dir, 'sentiment_results_*.json'))
    
    # Sort by modification time and get the latest
    latest_transcript = max(transcript_files, key=os.path.getmtime) if transcript_files else None
    latest_sentiment = max(sentiment_files, key=os.path.getmtime) if sentiment_files else None
    
    return latest_transcript, latest_sentiment

def read_file_content(file_path):
    """Read the content of a file."""
    if not file_path or not os.path.exists(file_path):
        return None
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read()

def analyze_sentiment(transcript_content, sentiment_data):
    """Use OpenAI to analyze the sentiment and provide a summary."""
    if not transcript_content or not sentiment_data:
        return "Error: Missing transcript or sentiment data"
    
    # Get API key
    api_key = load_api_keys()
    if not api_key:
        return "Error: Could not load OpenAI API key"
    
    # Prepare the prompt for OpenAI
    prompt = f"""
    Analyze the following conversation transcript and sentiment data to provide a brief summary of the overall sentiment.
    
    Conversation Transcript:
    {transcript_content}
    
    Sentiment Data:
    {json.dumps(sentiment_data, indent=2)}
    
    Please provide a concise summary that includes:
    1. The main emotional patterns observed
    2. Any significant shifts in sentiment
    3. The overall emotional tone of the conversation
    4. Any notable patterns or insights
    
    Keep the summary clear and professional.
    """
    
    try:
        # Initialize OpenAI client
        client = openai.OpenAI(api_key=api_key)
        
        # Call the API
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a professional sentiment analyst. Provide clear, concise, and insightful analysis of conversation sentiment."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=500
        )
        
        return response.choices[0].message.content.strip()
    
    except Exception as e:
        return f"Error in sentiment analysis: {str(e)}"

def main():
    # Get the transcripts directory
    transcripts_dir = os.path.join(os.path.dirname(__file__), 'gpt_transcripts')
    
    # Get latest files
    latest_transcript, latest_sentiment = get_latest_files(transcripts_dir)
    
    if not latest_transcript or not latest_sentiment:
        print("Error: Could not find latest transcript or sentiment files")
        return
    
    print(f"Analyzing latest files:")
    print(f"Transcript: {os.path.basename(latest_transcript)}")
    print(f"Sentiment: {os.path.basename(latest_sentiment)}")
    print("\n" + "="*50 + "\n")
    
    # Read file contents
    transcript_content = read_file_content(latest_transcript)
    sentiment_content = read_file_content(latest_sentiment)
    
    if not transcript_content or not sentiment_content:
        print("Error: Could not read file contents")
        return
    
    # Parse sentiment data
    try:
        sentiment_data = json.loads(sentiment_content)
    except json.JSONDecodeError:
        print("Error: Invalid JSON in sentiment file")
        return
    
    # Perform analysis
    analysis = analyze_sentiment(transcript_content, sentiment_data)
    
    # Print results
    print("Sentiment Analysis Summary:")
    print("="*50)
    print(analysis)
    print("="*50)

if __name__ == "__main__":
    main() 