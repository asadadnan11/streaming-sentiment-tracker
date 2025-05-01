import praw
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re
from datetime import datetime, timedelta
import nltk
import sys

# Download required NLTK data
nltk.download('vader_lexicon')
nltk.download('stopwords')
nltk.download('punkt')

# Set visualization style
sns.set_theme(style="whitegrid")  # Use seaborn's default theme
plt.rcParams['figure.figsize'] = [12, 6]
plt.rcParams['figure.dpi'] = 100

def main():
    try:
        # Initialize Reddit API client
        print("Initializing Reddit API client...")
        reddit = praw.Reddit(
            client_id='YOsFiClXBVfQGPNF8w2M6g',
            client_secret='7xm7tx0UkFWr-dUOGZM6j8bQBpP-Ow',
            user_agent='script:StreamingSentiment:v1.0',
            username='asadadnan_11',
            password='Eternity.9081'
        )
        
        # Test the connection
        print("Testing Reddit API connection...")
        test_subreddit = reddit.subreddit('test')
        print(f"Authenticated as: {reddit.user.me()}")
        print("Successfully connected to Reddit API!")
        
    except Exception as e:
        print("\nError: Could not connect to Reddit API. Please check your credentials.")
        print("Make sure you have:")
        print("1. Created a Reddit application at https://www.reddit.com/prefs/apps")
        print("2. Used the correct client_id and client_secret")
        print("3. Verified your Reddit account email")
        print(f"\nError details: {str(e)}")
        sys.exit(1)

    # Initialize VADER sentiment analyzer
    sia = SentimentIntensityAnalyzer()

    # Define streaming platforms to analyze
    platforms = ['Tubi', 'Netflix', 'Hulu', 'Disney+', 'HBO Max', 'Prime Video']

    # Function to collect posts about a specific platform
    def collect_posts(platform, limit=100):
        posts = []
        
        # Search in relevant subreddits
        subreddits = ['cordcutters', 'television', 'movies', 'streaming']
        
        for subreddit in subreddits:
            try:
                # Search for posts containing the platform name
                search_results = reddit.subreddit(subreddit).search(
                    f'{platform} streaming',
                    limit=limit,
                    time_filter='month'  # Get posts from the last month
                )
                
                for post in search_results:
                    posts.append({
                        'platform': platform,
                        'title': post.title,
                        'text': post.selftext,
                        'score': post.score,
                        'created_utc': datetime.fromtimestamp(post.created_utc),
                        'subreddit': subreddit
                    })
            except Exception as e:
                print(f"Error collecting posts for {platform} in r/{subreddit}: {str(e)}")
        
        return posts

    # Function to clean text
    def clean_text(text):
        if not isinstance(text, str):
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove special characters and numbers
        text = re.sub(r'[^\w\s]', '', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text

    # Function to get sentiment scores
    def get_sentiment(text):
        scores = sia.polarity_scores(text)
        return scores['compound']  # Using compound score for overall sentiment

    # Function to categorize sentiment
    def categorize_sentiment(score):
        if score >= 0.05:
            return 'Positive'
        elif score <= -0.05:
            return 'Negative'
        else:
            return 'Neutral'

    print("Collecting Reddit posts...")
    # Collect posts for all platforms
    all_posts = []
    for platform in platforms:
        print(f"Collecting posts for {platform}...")
        platform_posts = collect_posts(platform)
        all_posts.extend(platform_posts)
        print(f"Collected {len(platform_posts)} posts for {platform}")

    # Convert to DataFrame
    df = pd.DataFrame(all_posts)
    print(f"\nTotal posts collected: {len(df)}")

    # Clean and preprocess text
    print("\nCleaning and preprocessing text...")
    df['clean_title'] = df['title'].apply(clean_text)
    df['clean_text'] = df['text'].apply(clean_text)
    df['combined_text'] = df['clean_title'] + ' ' + df['clean_text']

    # Perform sentiment analysis
    print("\nPerforming sentiment analysis...")
    df['sentiment_score'] = df['combined_text'].apply(get_sentiment)
    df['sentiment'] = df['sentiment_score'].apply(categorize_sentiment)

    # Add date-based features
    df['date'] = df['created_utc'].dt.date
    df['week'] = df['created_utc'].dt.isocalendar().week

    # Display sentiment distribution
    print("\nSentiment Distribution:")
    print(df['sentiment'].value_counts(normalize=True).round(2))

    # Create visualizations
    print("\nCreating visualizations...")
    
    # 1. Overall sentiment distribution by platform
    plt.figure(figsize=(12, 6))
    sentiment_by_platform = pd.crosstab(df['platform'], df['sentiment'], normalize='index')
    sentiment_by_platform.plot(kind='bar', stacked=True)
    plt.title('Sentiment Distribution by Streaming Platform')
    plt.xlabel('Platform')
    plt.ylabel('Proportion of Posts')
    plt.legend(title='Sentiment')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('sentiment_by_platform.png')
    plt.close()

    # 2. Sentiment scores distribution
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='platform', y='sentiment_score', data=df)
    plt.title('Sentiment Score Distribution by Platform')
    plt.xlabel('Platform')
    plt.ylabel('Sentiment Score')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('sentiment_scores.png')
    plt.close()

    # 3. Sentiment trends over time
    plt.figure(figsize=(15, 8))
    
    # Calculate daily average sentiment for each platform
    daily_sentiment = df.groupby(['date', 'platform'])['sentiment_score'].mean().unstack()
    
    # Plot sentiment trends
    for platform in platforms:
        if platform in daily_sentiment.columns:
            plt.plot(daily_sentiment.index, daily_sentiment[platform], label=platform, marker='o')
    
    plt.title('Sentiment Trends Over Time by Platform')
    plt.xlabel('Date')
    plt.ylabel('Average Sentiment Score')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('sentiment_trends.png', bbox_inches='tight')
    plt.close()

    # 4. Weekly sentiment heatmap
    plt.figure(figsize=(12, 8))
    weekly_sentiment = df.groupby(['week', 'platform'])['sentiment_score'].mean().unstack()
    sns.heatmap(weekly_sentiment, cmap='RdYlGn', center=0, annot=True, fmt='.2f')
    plt.title('Weekly Sentiment Heatmap by Platform')
    plt.xlabel('Platform')
    plt.ylabel('Week Number')
    plt.tight_layout()
    plt.savefig('weekly_sentiment_heatmap.png')
    plt.close()

    print("\nAnalysis complete! Check the generated PNG files for visualizations:")
    print("1. sentiment_by_platform.png - Overall sentiment distribution")
    print("2. sentiment_scores.png - Distribution of sentiment scores")
    print("3. sentiment_trends.png - Daily sentiment trends over time")
    print("4. weekly_sentiment_heatmap.png - Weekly sentiment patterns")

if __name__ == "__main__":
    main() 