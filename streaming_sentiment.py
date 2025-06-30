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

# Download NLTK stuff - this took me forever to figure out
nltk.download('vader_lexicon')  # for sentiment analysis
nltk.download('stopwords')  # probably don't need this one but just in case
nltk.download('punkt')

# Make the plots look decent
sns.set_theme(style="whitegrid")  # seaborn's default theme looks good
plt.rcParams['figure.figsize'] = [12, 6]
plt.rcParams['figure.dpi'] = 100

def main():
    try:
        # Set up Reddit API connection
        print("Initializing Reddit API client...")
        reddit_client = praw.Reddit(
            client_id='YOsFiClXBVfQGPNF8w2M6g',
            client_secret='7xm7tx0UkFWr-dUOGZM6j8bQBpP-Ow',
            user_agent='script:StreamingSentiment:v1.0',
            username='asadadnan_11',
            password='Eternity.9081'
        )
        
        # Test if the connection works
        print("Testing Reddit API connection...")
        test_sub = reddit_client.subreddit('test')
        print(f"Authenticated as: {reddit_client.user.me()}")
        print("Successfully connected to Reddit API!")
        
    except Exception as e:
        print("\nError: Could not connect to Reddit API. Please check your credentials.")
        print("Make sure you have:")
        print("1. Created a Reddit application at https://www.reddit.com/prefs/apps")
        print("2. Used the correct client_id and client_secret")
        print("3. Verified your Reddit account email")
        print(f"\nError details: {str(e)}")
        sys.exit(1)

    # Set up VADER sentiment analyzer
    sentiment_analyzer = SentimentIntensityAnalyzer()

    # List of streaming platforms to analyze
    streaming_platforms = ['Tubi', 'Netflix', 'Hulu', 'Disney+', 'HBO Max', 'Prime Video']

    # Function to collect posts about streaming platforms
    def collect_posts(platform_name, limit=100):
        posts_data = []
        
        # Subreddits where people talk about streaming
        subreddit_list = ['cordcutters', 'television', 'movies', 'streaming']
        
        for sub_name in subreddit_list:
            try:
                # Search for posts mentioning the platform
                search_results = reddit_client.subreddit(sub_name).search(
                    f'{platform_name} streaming',
                    limit=limit,
                    time_filter='month'  # last month's posts - tried 'week' but too few results
                )
                
                for post in search_results:
                    posts_data.append({
                        'platform': platform_name,
                        'title': post.title,
                        'text': post.selftext,
                        'score': post.score,
                        'created_utc': datetime.fromtimestamp(post.created_utc),
                        'subreddit': sub_name
                    })
            except Exception as e:
                print(f"Error collecting posts for {platform_name} in r/{sub_name}: {str(e)}")
        
        return posts_data

    # Clean up text data
    def clean_text(input_text):
        if not isinstance(input_text, str):
            return ""
        
        # Make it lowercase
        text = input_text.lower()
        
        # Get rid of URLs - they don't help with sentiment
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove special characters and numbers
        text = re.sub(r'[^\w\s]', '', text)
        
        # Clean up whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text

    # Get sentiment score from text
    def get_sentiment(text_input):
        sentiment_scores = sentiment_analyzer.polarity_scores(text_input)
        return sentiment_scores['compound']  # compound score works best

    # Convert sentiment score to category
    def categorize_sentiment(sentiment_score):
        if sentiment_score >= 0.05:
            return 'Positive'
        elif sentiment_score <= -0.05:
            return 'Negative'
        else:
            return 'Neutral'

    print("Collecting Reddit posts...")
    # Get posts for all platforms
    all_posts_data = []
    for platform in streaming_platforms:
        print(f"Collecting posts for {platform}...")
        platform_posts = collect_posts(platform)
        all_posts_data.extend(platform_posts)
        print(f"Collected {len(platform_posts)} posts for {platform}")

    # Convert to pandas DataFrame
    posts_df = pd.DataFrame(all_posts_data)
    print(f"\nTotal posts collected: {len(posts_df)}")

    # Clean and process the text data
    print("\nCleaning and preprocessing text...")
    posts_df['clean_title'] = posts_df['title'].apply(clean_text)
    posts_df['clean_text'] = posts_df['text'].apply(clean_text)
    posts_df['combined_text'] = posts_df['clean_title'] + ' ' + posts_df['clean_text']

    # Run sentiment analysis
    print("\nPerforming sentiment analysis...")
    posts_df['sentiment_score'] = posts_df['combined_text'].apply(get_sentiment)
    posts_df['sentiment'] = posts_df['sentiment_score'].apply(categorize_sentiment)

    # Add some date-based columns
    posts_df['date'] = posts_df['created_utc'].dt.date
    posts_df['week'] = posts_df['created_utc'].dt.isocalendar().week

    # Show sentiment breakdown
    print("\nSentiment Distribution:")
    print(posts_df['sentiment'].value_counts(normalize=True).round(2))

    # Create some visualizations
    print("\nCreating visualizations...")
    
    # 1. Sentiment distribution by platform
    plt.figure(figsize=(12, 6))
    sentiment_by_platform = pd.crosstab(posts_df['platform'], posts_df['sentiment'], normalize='index')
    sentiment_by_platform.plot(kind='bar', stacked=True)
    plt.title('Sentiment Distribution by Streaming Platform')
    plt.xlabel('Platform')
    plt.ylabel('Proportion of Posts')
    plt.legend(title='Sentiment')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('sentiment_by_platform.png')
    plt.close()

    # 2. Box plot of sentiment scores
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='platform', y='sentiment_score', data=posts_df)
    plt.title('Sentiment Score Distribution by Platform')
    plt.xlabel('Platform')
    plt.ylabel('Sentiment Score')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('sentiment_scores.png')
    plt.close()

    # 3. Sentiment trends over time
    plt.figure(figsize=(15, 8))
    
    # Calculate daily averages for each platform
    daily_sentiment_data = posts_df.groupby(['date', 'platform'])['sentiment_score'].mean().unstack()
    
    # Plot the trends
    for platform in streaming_platforms:
        if platform in daily_sentiment_data.columns:
            plt.plot(daily_sentiment_data.index, daily_sentiment_data[platform], label=platform, marker='o')
    
    plt.title('Sentiment Trends Over Time by Platform')
    plt.xlabel('Date')
    plt.ylabel('Average Sentiment Score')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('sentiment_trends.png', bbox_inches='tight')
    plt.close()

    # 4. Weekly sentiment heatmap - this one was a bit tricky
    plt.figure(figsize=(12, 8))
    weekly_sentiment_data = posts_df.groupby(['week', 'platform'])['sentiment_score'].mean().unstack()
    sns.heatmap(weekly_sentiment_data, cmap='RdYlGn', center=0, annot=True, fmt='.2f')
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