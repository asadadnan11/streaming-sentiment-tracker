import streamlit as st
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
from dateutil.relativedelta import relativedelta
import random

# Download required NLTK data
@st.cache_resource
def download_nltk_data():
    nltk.download('vader_lexicon')
    nltk.download('stopwords')
    nltk.download('punkt')

# Initialize NLTK
download_nltk_data()

# Set visualization style
sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = [12, 6]
plt.rcParams['figure.dpi'] = 100

# Initialize Reddit API client
@st.cache_resource
def init_reddit():
    return praw.Reddit(
        client_id='YOsFiClXBVfQGPNF8w2M6g',
        client_secret='7xm7tx0UkFWr-dUOGZM6j8bQBpP-Ow',
        user_agent='script:StreamingSentiment:v1.0',
        username='asadadnan_11',
        password='Eternity.9081'
    )

# Function to collect posts about a specific topic
def collect_posts(reddit, topic, limit=100, analysis_type='streaming'):
    posts = []
    
    if analysis_type == 'streaming':
        subreddits = [
            'television', 'movies', 'streaming',
            'netflix', 'hulu', 'cordcutters',
            'PrimeVideo', 'BestofStreamingVideo'
        ]
        search_query = topic
    elif analysis_type == 'director':
        subreddits = [
            'movies', 'TrueFilm', 'criterion',
            'MovieSuggestions', 'flicks', 'classicfilms',
            'FilmDiscussion'
        ]
        # For directors, search both with and without "director" keyword
        search_query = f'"{topic}"'  # Exact match search
    else:  # country
        subreddits = [
            'movies', 'TrueFilm', 'criterion',
            'MovieSuggestions', 'flicks', 'ForeignMovies',
            'classicfilms', 'FilmDiscussion'
        ]
        search_query = f'{topic} film'
    
    # Calculate start date (6 months ago)
    end_date = datetime.now()
    start_date = end_date - relativedelta(months=6)
    
    total_attempts = len(subreddits)
    successful_attempts = 0
    total_posts = 0  # Track total posts collected
    
    for subreddit in subreddits:
        if total_posts >= limit:  # Stop if we've reached the limit
            break
            
        try:
            # First verify the subreddit exists and is accessible
            sub = reddit.subreddit(subreddit)
            try:
                sub.created_utc
            except Exception:
                continue
                
            search_results = list(sub.search(
                search_query,
                limit=limit,  # Reduced from limit*2 to just limit
                time_filter='year'
            ))
            
            # Shuffle the results to get a random sample from each subreddit
            random.shuffle(search_results)
            
            post_count = 0
            for post in search_results:
                if total_posts >= limit:  # Check global limit
                    break
                    
                post_date = datetime.fromtimestamp(post.created_utc)
                if start_date <= post_date <= end_date:
                    posts.append({
                        'topic': topic,
                        'title': post.title,
                        'text': post.selftext,
                        'score': post.score,
                        'created_utc': post_date,
                        'subreddit': subreddit,
                        'analysis_type': analysis_type
                    })
                    post_count += 1
                    total_posts += 1
            
            if post_count > 0:
                successful_attempts += 1
                
        except Exception as e:
            continue
    
    if successful_attempts == 0:
        st.warning(f"No data found for {topic}. Tried {total_attempts} subreddits but all failed.")
    else:
        st.success(f"Successfully collected {len(posts)} posts for {topic} from {successful_attempts} subreddits.")
    
    # Ensure we don't return more than the limit
    return posts[:limit]

# Function to clean text
def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Function to get sentiment scores
@st.cache_data
def get_sentiment(text):
    sia = SentimentIntensityAnalyzer()
    scores = sia.polarity_scores(text)
    return scores['compound']

# Function to categorize sentiment
def categorize_sentiment(score):
    if score >= 0.05:
        return 'Positive'
    elif score <= -0.05:
        return 'Negative'
    else:
        return 'Neutral'

# Function to create sentiment distribution plot
def plot_sentiment_distribution(df, analysis_type):
    fig, ax = plt.subplots(figsize=(12, 6))
    sentiment_by_topic = pd.crosstab(df['topic'], df['sentiment'], normalize='index')
    sentiment_by_topic.plot(kind='bar', stacked=True, ax=ax)
    plt.title(f'Sentiment Distribution by {analysis_type}')
    plt.xlabel(analysis_type)
    plt.ylabel('Proportion of Posts')
    plt.legend(title='Sentiment')
    plt.xticks(rotation=45)
    plt.tight_layout()
    return fig

# Function to create sentiment scores plot
def plot_sentiment_scores(df, analysis_type):
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(x='topic', y='sentiment_score', data=df, ax=ax)
    plt.title(f'Sentiment Score Distribution by {analysis_type}')
    plt.xlabel(analysis_type)
    plt.ylabel('Sentiment Score')
    plt.xticks(rotation=45)
    plt.tight_layout()
    return fig

# Function to create monthly sentiment trends plot
def plot_monthly_trends(df, topics, analysis_type):
    # Convert datetime to month-year format
    df['month'] = df['created_utc'].dt.to_period('M')
    
    # Calculate monthly averages
    monthly_sentiment = df.groupby(['month', 'topic'])['sentiment_score'].mean().unstack()
    
    fig, ax = plt.subplots(figsize=(15, 8))
    
    for topic in topics:
        if topic in monthly_sentiment.columns:
            plt.plot(range(len(monthly_sentiment.index)), 
                    monthly_sentiment[topic], 
                    label=topic, 
                    marker='o')
    
    # Set x-axis labels to month names
    plt.xticks(range(len(monthly_sentiment.index)), 
               [str(x) for x in monthly_sentiment.index], 
               rotation=45)
    
    plt.title(f'Monthly Sentiment Trends by {analysis_type}')
    plt.xlabel('Month')
    plt.ylabel('Average Sentiment Score')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)
    plt.tight_layout()
    return fig

# Function to create weekly sentiment heatmap
def plot_weekly_heatmap(df, analysis_type):
    fig, ax = plt.subplots(figsize=(12, 8))
    weekly_sentiment = df.groupby(['week', 'topic'])['sentiment_score'].mean().unstack()
    sns.heatmap(weekly_sentiment, cmap='RdYlGn', center=0, annot=True, fmt='.2f', ax=ax)
    plt.title(f'Weekly Sentiment Patterns by {analysis_type}')
    plt.xlabel(analysis_type)
    plt.ylabel('Week Number')
    plt.tight_layout()
    return fig

# Main Streamlit app
def main():
    st.title('Cinema Analysis')
    st.write('Analyzing Reddit sentiment about streaming platforms, directors, and international cinema')

    # Sidebar controls
    st.sidebar.header('Analysis Controls')
    
    # Analysis type selection
    analysis_type = st.sidebar.radio(
        'Select Analysis Type',
        ['Streaming Platforms', 'Directors', 'International Cinema']
    )
    
    if analysis_type == 'Streaming Platforms':
        platforms = [
            'Netflix', 'Hulu', 'Disney+', 'HBO Max', 'Prime Video',
            'Apple TV+', 'Paramount+', 'Peacock', 'Tubi', 'Criterion Channel',
            'MUBI', 'Shudder', 'Crackle', 'Pluto TV', 'YouTube Premium'
        ]
        selected_topics = st.sidebar.multiselect(
            'Select Platforms to Analyze',
            platforms,
            default=['Netflix', 'Hulu', 'Disney+']
        )
        display_type = "Platform"
    elif analysis_type == 'Directors':
        directors = [
            'Bernardo Bertolucci', 'Martin Scorsese', 'Satyajit Ray', 'Jean-Luc Godard',
            'Andrei Tarkovsky', 'Wim Wenders', 'Denis Villeneuve', 'Abbas Kiarostami',
            'Majid Majidi', 'Terrence Malick', 'Greta Gerwig', 'Christopher Nolan',
            'Michael Mann', 'Wong Kar-wai', 'Akira Kurosawa', 'Ingmar Bergman',
            'Federico Fellini', 'Alfred Hitchcock', 'Stanley Kubrick', 'David Lynch',
            'Hayao Miyazaki', 'Park Chan-wook', 'Bong Joon-ho', 'Pedro Almodóvar',
            'Lars von Trier', 'David Fincher', 'Quentin Tarantino', 'Coen Brothers',
            'Wes Anderson', 'Guillermo del Toro'
        ]
        
        # Custom director input with better validation
        custom_director = st.sidebar.text_input('Or enter a custom director name:').strip()
        if custom_director and custom_director not in directors and len(custom_director) > 2:
            # Add custom director to the list for selection
            directors = sorted(directors + [custom_director])
            st.sidebar.success(f"Added {custom_director} to the list!")
        
        selected_topics = st.sidebar.multiselect(
            'Select Directors to Analyze',
            directors,
            default=['Martin Scorsese', 'Christopher Nolan', 'Denis Villeneuve']
        )
        display_type = "Director"
    else:
        countries = [
            'France', 'Italy', 'Japan', 'South Korea', 'India', 'Iran',
            'Germany', 'Sweden', 'Russia', 'Spain', 'Mexico', 'Brazil',
            'China', 'Hong Kong', 'Taiwan', 'Thailand', 'Turkey', 'Egypt',
            'Argentina', 'Denmark', 'Poland', 'Czech Republic', 'Hungary',
            'Romania', 'Greece', 'Belgium', 'Netherlands', 'Norway',
            'Finland', 'Austria', 'Vietnam', 'Philippines', 'Indonesia',
            'Chile', 'Colombia', 'Venezuela', 'Morocco', 'Nigeria',
            'South Africa', 'Israel', 'Lebanon'
        ]
        
        # Custom country input with better validation
        custom_country = st.sidebar.text_input('Or enter a custom country name:').strip()
        if custom_country and custom_country not in countries and len(custom_country) > 2:
            # Add custom country to the list for selection
            countries = sorted(countries + [custom_country])
            st.sidebar.success(f"Added {custom_country} to the list!")
        
        selected_topics = st.sidebar.multiselect(
            'Select Countries to Analyze',
            countries,
            default=['France', 'Japan', 'South Korea']
        )
        display_type = "Country"

    # Posts per topic
    posts_limit = st.sidebar.slider(
        'Posts per Topic',
        min_value=10,
        max_value=200,
        value=100,
        step=10
    )

    # Initialize Reddit client
    try:
        reddit = init_reddit()
        
        if st.button('Run Analysis'):
            with st.spinner('Collecting and analyzing Reddit posts...'):
                # Collect and process data
                all_posts = []
                progress_bar = st.progress(0)
                
                for i, topic in enumerate(selected_topics):
                    st.write(f"Collecting posts for {topic}...")
                    topic_posts = collect_posts(
                        reddit, 
                        topic, 
                        limit=posts_limit,
                        analysis_type='streaming' if analysis_type == 'Streaming Platforms' else 
                                    'director' if analysis_type == 'Directors' else 'country'
                    )
                    all_posts.extend(topic_posts)
                    progress = (i + 1) / len(selected_topics)
                    progress_bar.progress(progress)
                
                # Create DataFrame
                df = pd.DataFrame(all_posts)
                
                if len(df) > 0:
                    # Process data
                    df['clean_title'] = df['title'].apply(clean_text)
                    df['clean_text'] = df['text'].apply(clean_text)
                    df['combined_text'] = df['clean_title'] + ' ' + df['clean_text']
                    df['sentiment_score'] = df['combined_text'].apply(get_sentiment)
                    df['sentiment'] = df['sentiment_score'].apply(categorize_sentiment)
                    df['date'] = df['created_utc'].dt.date
                    df['week'] = df['created_utc'].dt.isocalendar().week

                    # Display results in tabs
                    tab1, tab2, tab3, tab4, tab5 = st.tabs([
                        "Overview", 
                        "Sentiment Distribution", 
                        "Sentiment Scores", 
                        "Time Trends",
                        "Weekly Patterns"
                    ])

                    with tab1:
                        st.header("Analysis Overview")
                        st.write(f"Total posts analyzed: {len(df)}")
                        
                        # Display sentiment distribution
                        st.subheader("Overall Sentiment Distribution")
                        sentiment_counts = df['sentiment'].value_counts(normalize=True).round(3)
                        col1, col2, col3 = st.columns(3)
                        col1.metric("Positive", f"{sentiment_counts.get('Positive', 0):.1%}")
                        col2.metric("Neutral", f"{sentiment_counts.get('Neutral', 0):.1%}")
                        col3.metric("Negative", f"{sentiment_counts.get('Negative', 0):.1%}")

                        # Display posts per topic
                        st.subheader(f"Posts per {display_type}")
                        topic_counts = df['topic'].value_counts()
                        st.bar_chart(topic_counts)

                    with tab2:
                        st.header(f"Sentiment Distribution by {display_type}")
                        st.pyplot(plot_sentiment_distribution(df, display_type))

                    with tab3:
                        st.header(f"Sentiment Score Distribution by {display_type}")
                        st.pyplot(plot_sentiment_scores(df, display_type))

                    with tab4:
                        st.header("Time Trends")
                        if analysis_type == 'Directors':
                            st.info("Temporal analysis is not shown for directors as it may not be meaningful for historical figures.")
                        else:
                            st.pyplot(plot_monthly_trends(df, selected_topics, display_type))

                    with tab5:
                        st.header("Weekly Patterns")
                        if analysis_type == 'Directors' or analysis_type == 'International Cinema':
                            st.info("Weekly patterns are not shown for directors and international cinema as they may not be meaningful.")
                        else:
                            st.pyplot(plot_weekly_heatmap(df, display_type))

                    # Show raw data option
                    if st.checkbox('Show Raw Data'):
                        st.subheader('Raw Data')
                        st.dataframe(df[['topic', 'title', 'sentiment', 'sentiment_score', 'created_utc', 'subreddit']])
                
                else:
                    st.error("No posts found for the selected topics.")
                
    except Exception as e:
        st.error(f"Error connecting to Reddit API: {str(e)}")

if __name__ == "__main__":
    main() 