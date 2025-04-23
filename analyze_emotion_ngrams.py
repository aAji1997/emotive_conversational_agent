import json
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
from nltk import ngrams
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import numpy as np
import os
from pathlib import Path
from wordcloud import WordCloud
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from scipy.stats import pearsonr

# Define core emotions with their colors
EMOTION_COLORS = {
    "Joy": "#FFD700",  # Gold
    "Trust": "#4169E1",  # Royal Blue
    "Fear": "#800000",  # Maroon
    "Surprise": "#FFA500",  # Orange
    "Sadness": "#4682B4",  # Steel Blue
    "Disgust": "#228B22",  # Forest Green
    "Anger": "#FF4500",  # Orange Red
    "Anticipation": "#9370DB"  # Medium Purple
}

# Define core emotions
CORE_EMOTIONS = list(EMOTION_COLORS.keys())

# Set the style for all plots
plt.style.use('seaborn-v0_8')  # Using a specific seaborn style version
sns.set_theme(style="whitegrid")  # Set seaborn theme

# Download required NLTK data with error handling
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    print("Downloading NLTK punkt data...")
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    print("Downloading NLTK stopwords data...")
    nltk.download('stopwords')

try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    print("Downloading NLTK punkt_tab data...")
    nltk.download('punkt_tab')

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    print("Downloading NLTK wordnet data...")
    nltk.download('wordnet')

def get_user_folders():
    """Get list of user folders in gpt_transcripts directory."""
    transcripts_dir = Path('gpt_realtime/gpt_transcripts')
    if not transcripts_dir.exists():
        print("Error: gpt_transcripts directory not found.")
        return None
        
    # Get all subdirectories (user folders)
    user_folders = [f for f in transcripts_dir.iterdir() if f.is_dir()]
    if not user_folders:
        print("Error: No user folders found in gpt_transcripts directory.")
        return None
        
    return user_folders

def select_user_folder():
    """Let user enter their username to find their folder."""
    user_folders = get_user_folders()
    if not user_folders:
        return None
        
    while True:
        username = input("\nEnter your username: ").strip()
        if not username:
            print("Username cannot be empty. Please try again.")
            continue
            
        # Find folder matching the username
        matching_folder = next((f for f in user_folders if f.name.lower() == username.lower()), None)
        if matching_folder:
            print(f"\nFound folder for user: {username}")
            return matching_folder
            
        print(f"\nNo folder found for username: {username}")
        print("Available usernames:")
        for folder in user_folders:
            print(f"- {folder.name}")
        print("\nPlease try again with one of the available usernames.")

def load_sentiment_data(user_folder):
    """Load sentiment analysis results and conversation transcripts for a specific user."""
    # Get latest sentiment file
    sentiment_files = list(user_folder.glob('sentiment_results_*.json'))
    if not sentiment_files:
        print(f"Error: No sentiment results files found in {user_folder} directory.")
        return None
    
    # Get latest transcript file
    transcript_files = list(user_folder.glob('conversation_*.txt'))
    if not transcript_files:
        print(f"Error: No conversation transcript files found in {user_folder} directory.")
        return None
    
    # Get the most recent files
    latest_sentiment = max(sentiment_files, key=os.path.getmtime)
    latest_transcript = max(transcript_files, key=os.path.getmtime)
    
    print(f"\nUsing latest sentiment file: {latest_sentiment.name}")
    print(f"Using latest transcript file: {latest_transcript.name}")
    
    try:
        # Load sentiment data
        with open(latest_sentiment, 'r') as f:
            sentiment_data = json.load(f)
        
        # Load transcript
        with open(latest_transcript, 'r', encoding='utf-8') as f:
            transcript = f.read()
        
        # Split transcript into messages
        messages = []
        for line in transcript.split('\n'):
            if line.startswith('User:') or line.startswith('Assistant:'):
                messages.append(line)
        
        # Match sentiment data with messages
        for entry in sentiment_data:
            if len(messages) > 0:
                entry['text'] = messages.pop(0)
        
        return sentiment_data
        
    except FileNotFoundError:
        print(f"Error: Could not find required files.")
        return None
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON in {latest_sentiment}")
        return None

def extract_ngrams(text, n=3):
    """Extract n-grams from text, removing stopwords and lemmatizing words."""
    # Initialize lemmatizer
    lemmatizer = WordNetLemmatizer()
    
    # Get standard stop words
    stop_words = set(stopwords.words('english'))
    
    # Add 'samantha' to the list of words to ignore
    stop_words.add('samantha')
    
    # Tokenize and lemmatize
    words = nltk.word_tokenize(text.lower())
    words = [lemmatizer.lemmatize(word) for word in words if word.isalpha() and word not in stop_words]
    
    return [' '.join(gram) for gram in ngrams(words, n)]

def analyze_emotion_ngrams(sentiment_data):
    """Analyze 3-grams for each emotion category."""
    emotion_ngrams = {emotion: Counter() for emotion in CORE_EMOTIONS}
    emotion_scores = {emotion: [] for emotion in CORE_EMOTIONS}
    emotion_vectors = {emotion: [] for emotion in CORE_EMOTIONS}  # For correlation analysis
    
    print("\nProcessing sentiment data:")
    print(f"Number of entries: {len(sentiment_data)}")
    
    for entry in sentiment_data:
        text = entry.get('text', '')
        entry_scores = entry.get('emotion_scores', {})
        
        print(f"\nText: {text[:100]}...")  # Print first 100 chars of text
        print(f"Emotion scores: {entry_scores}")
        
        # Get trigrams from text
        text_ngrams = extract_ngrams(text)
        print(f"Extracted trigrams: {text_ngrams[:5]}...")  # Print first 5 trigrams
        
        # Associate n-grams with their emotion scores
        for emotion, score in entry_scores.items():
            if score >= 1.0:  # Lower threshold to capture more emotional content
                print(f"Found high score for {emotion}: {score}")
                emotion_scores[emotion].append(score)
                emotion_vectors[emotion].append(1)  # Mark presence of emotion
                for ngram in text_ngrams:
                    emotion_ngrams[emotion][ngram] += 1
            else:
                emotion_vectors[emotion].append(0)  # Mark absence of emotion
    
    print("\nEmotion n-grams collected:")
    for emotion, counter in emotion_ngrams.items():
        if counter:
            print(f"\n{emotion} - Trigrams:")
            print(f"Unique trigrams: {len(counter)}")
            print(f"Top 3 trigrams: {counter.most_common(3)}")
    
    return emotion_ngrams, emotion_scores, emotion_vectors

def calculate_emotion_correlations(emotion_vectors):
    """Calculate correlation matrix and p-values for emotion vectors."""
    df = pd.DataFrame(emotion_vectors)
    
    # Check if we have enough data points
    if len(df) < 2:
        print("\nWarning: Not enough data points to calculate correlations (minimum 2 required)")
        return None, None
    
    # Calculate correlation matrix
    corr_matrix = df.corr()
    
    # Calculate p-values for significance
    p_values = pd.DataFrame(index=corr_matrix.index, columns=corr_matrix.columns)
    for i in corr_matrix.index:
        for j in corr_matrix.columns:
            if i != j:
                corr, p_value = pearsonr(df[i], df[j])
                p_values.loc[i, j] = p_value
            else:
                p_values.loc[i, j] = 0
    
    return corr_matrix, p_values

def plot_emotion_correlations(corr_matrix, p_values, output_dir):
    """Create a heatmap of emotion correlations."""
    if corr_matrix is None or p_values is None:
        print("Skipping correlation plot due to insufficient data")
        return
        
    plt.figure(figsize=(12, 10))
    
    # Create mask for significant correlations (p < 0.05)
    mask = p_values > 0.05
    
    # Create heatmap
    sns.heatmap(corr_matrix, 
                annot=True, 
                cmap='coolwarm', 
                center=0,
                mask=mask,
                fmt='.2f',
                square=True,
                linewidths=.5,
                cbar_kws={'shrink': .8})
    
    plt.title('Emotion Correlations (Significant Only, p < 0.05)')
    plt.tight_layout()
    
    # Save plot
    output_file = output_dir / 'emotion_correlations.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved correlation heatmap to {output_file}")

def create_wordcloud(emotion_ngrams, emotion, output_dir, suffix=''):
    """Create a word cloud for the given emotion's n-grams."""
    if not emotion_ngrams[emotion]:
        return
    
    # Calculate uniqueness scores for each n-gram
    all_ngrams = set()
    for other_emotion in CORE_EMOTIONS:
        if other_emotion != emotion:
            all_ngrams.update(emotion_ngrams[other_emotion].keys())
    
    # Create a dictionary with uniqueness scores
    uniqueness_scores = {}
    for ngram, count in emotion_ngrams[emotion].items():
        # Calculate how unique this n-gram is to this emotion
        uniqueness = 1.0
        for other_emotion in CORE_EMOTIONS:
            if other_emotion != emotion:
                other_count = emotion_ngrams[other_emotion].get(ngram, 0)
                if other_count > 0:
                    uniqueness *= (1.0 / (other_count + 1))  # Add 1 to avoid division by zero
        
        # Combine frequency with uniqueness
        uniqueness_scores[ngram] = count * uniqueness
    
    # Create word cloud with uniqueness-weighted frequencies
    wordcloud = WordCloud(
        width=800,
        height=400,
        background_color='white',
        colormap=LinearSegmentedColormap.from_list('custom', ['#ffffff', EMOTION_COLORS[emotion]]),
        max_words=50,  # Reduced to show more distinctive words
        min_font_size=10,
        max_font_size=100,
        relative_scaling=0.5,  # Adjust relative scaling
        contour_width=1,
        contour_color=EMOTION_COLORS[emotion],
        prefer_horizontal=0.7,  # Prefer horizontal words
        collocations=False  # Don't show repeated words
    ).generate_from_frequencies(uniqueness_scores)
    
    # Plot word cloud
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title(f'Word Cloud for {emotion} (Uniqueness-Weighted)')
    
    # Add a subtitle with the most unique phrases
    top_unique = sorted(uniqueness_scores.items(), key=lambda x: x[1], reverse=True)[:3]
    if top_unique:
        subtitle = "Most unique phrases:\n" + "\n".join([f"- {phrase}" for phrase, _ in top_unique])
        plt.figtext(0.02, 0.02, subtitle, fontsize=8, bbox=dict(facecolor='white', alpha=0.8))
    
    # Save plot
    output_file = output_dir / f'{emotion.lower()}_wordcloud_{suffix}.png'
    plt.savefig(output_file, bbox_inches='tight', pad_inches=0.1)
    plt.close()
    print(f"Saved word cloud to {output_file}")

def plot_emotion_distribution(emotion_scores, output_dir):
    """Create a box plot showing the distribution of emotion scores."""
    # Prepare data for plotting
    data = []
    labels = []
    for emotion, scores in emotion_scores.items():
        if scores:  # Only include emotions with scores
            data.append(scores)
            labels.append(emotion)
    
    if not data:
        print("\nNo emotion scores to plot")
        return
        
    # Create box plot
    plt.figure(figsize=(12, 6))
    box = plt.boxplot(data, labels=labels, patch_artist=True)
    
    # Color the boxes
    for patch, emotion in zip(box['boxes'], labels):
        patch.set_facecolor(EMOTION_COLORS[emotion])
    
    plt.title('Distribution of Emotion Scores')
    plt.ylabel('Score')
    plt.xticks(rotation=45)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Save plot
    output_file = output_dir / 'emotion_distribution.png'
    plt.savefig(output_file, bbox_inches='tight')
    plt.close()
    print(f"Saved emotion distribution plot to {output_file}")

def plot_emotion_ngrams(emotion_ngrams, emotion_scores, user_folder, top_n=10):
    """Create visualization of top n-grams for each emotion."""
    plots_created = False
    
    # Create output directory for this user
    output_dir = Path('ngram_analysis') / user_folder.name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Calculate uniqueness scores for all n-grams
    uniqueness_scores = {}
    for emotion in CORE_EMOTIONS:
        uniqueness_scores[emotion] = {}
        for ngram, count in emotion_ngrams[emotion].items():
            # Calculate how unique this n-gram is to this emotion
            uniqueness = 1.0
            for other_emotion in CORE_EMOTIONS:
                if other_emotion != emotion:
                    other_count = emotion_ngrams[other_emotion].get(ngram, 0)
                    if other_count > 0:
                        uniqueness *= (1.0 / (other_count + 1))
            uniqueness_scores[emotion][ngram] = count * uniqueness
    
    for emotion, ngram_counter in emotion_ngrams.items():
        if not ngram_counter:  # Skip if no n-grams found
            print(f"\nNo n-grams found for {emotion}")
            continue
        
        # Get top n-grams by uniqueness score
        top_ngrams = sorted(uniqueness_scores[emotion].items(), key=lambda x: x[1], reverse=True)[:top_n]
        
        if not top_ngrams:  # Skip if no n-grams
            print(f"\nNo top n-grams found for {emotion}")
            continue
        
        print(f"\nCreating word cloud for {emotion}")
        print(f"Top unique n-grams: {top_ngrams}")
        
        # Create word cloud for this emotion
        create_wordcloud(emotion_ngrams, emotion, output_dir)
        plots_created = True
    
    if not plots_created:
        print("\nNo word clouds were created. Check if any emotions had scores >= 1.0")
    else:
        # Create emotion distribution plot
        plot_emotion_distribution(emotion_scores, output_dir)

def main():
    # Let user select which folder to analyze
    user_folder = select_user_folder()
    if not user_folder:
        return
        
    # Load sentiment data for the selected user
    sentiment_data = load_sentiment_data(user_folder)
    if not sentiment_data:
        return
        
    # Analyze n-grams
    emotion_ngrams, emotion_scores, emotion_vectors = analyze_emotion_ngrams(sentiment_data)
    
    # Calculate and plot emotion correlations
    corr_matrix, p_values = calculate_emotion_correlations(emotion_vectors)
    plot_emotion_correlations(corr_matrix, p_values, Path('ngram_analysis') / user_folder.name)
    
    # Create visualizations
    plot_emotion_ngrams(emotion_ngrams, emotion_scores, user_folder)
    
    print(f"\nAnalysis complete. Visualizations saved in ngram_analysis/{user_folder.name}/")

if __name__ == "__main__":
    main() 