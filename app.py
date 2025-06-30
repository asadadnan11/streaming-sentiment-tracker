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

# Download required NLTK data - had to figure this out the hard way
@st.cache_resource
def download_nltk_data():
    nltk.download('vader_lexicon')
    nltk.download('stopwords')
    nltk.download('punkt')  # TODO: might not need all of these but better safe than sorry

# Initialize NLTK stuff
download_nltk_data()

# Set up the plots to look decent
sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = [12, 6]
plt.rcParams['figure.dpi'] = 100  # makes it look crisp

# Reddit API setup - these are my actual credentials
@st.cache_resource
def init_reddit():
    reddit_client = praw.Reddit(
        client_id='YOsFiClXBVfQGPNF8w2M6g',
        client_secret='7xm7tx0UkFWr-dUOGZM6j8bQBpP-Ow',
        user_agent='script:StreamingSentiment:v1.0',
        username='asadadnan_11',
        password='Eternity.9081'
    )
    return reddit_client

# This function does the heavy lifting for collecting posts
def collect_posts(reddit, topic, limit=100, analysis_type='streaming'):
    posts_list = []
    
    # Different subreddits for different analysis types
    if analysis_type == 'streaming':
        subreddits_to_search = [
            'television', 'movies', 'streaming',
            'netflix', 'hulu', 'cordcutters',
            'PrimeVideo', 'BestofStreamingVideo'
        ]
        search_term = topic
    elif analysis_type == 'director':
        subreddits_to_search = [
            'movies', 'TrueFilm', 'criterion',
            'MovieSuggestions', 'flicks', 'classicfilms',
            'FilmDiscussion'
        ]
        # For directors, search both with and without "director" keyword
        search_term = f'"{topic}"'  # exact match works better
    else:  # country analysis
        subreddits_to_search = [
            'movies', 'TrueFilm', 'criterion',
            'MovieSuggestions', 'flicks', 'ForeignMovies',
            'classicfilms', 'FilmDiscussion'
        ]
        search_term = f'{topic} film'
    
    # Get posts from last 6 months
    end_date = datetime.now()
    start_date = end_date - relativedelta(months=6)
    
    total_attempts = len(subreddits_to_search)
    successful_attempts = 0
    total_posts_collected = 0  # keep track of how many we actually got
    
    for subreddit_name in subreddits_to_search:
        if total_posts_collected >= limit:  # don't go over the limit
            break
            
        try:
            # Check if subreddit exists first
            sub = reddit.subreddit(subreddit_name)
            try:
                sub.created_utc  # this will fail if subreddit doesn't exist
            except Exception:
                continue
                
            search_results = list(sub.search(
                search_term,
                limit=limit,  # used to be limit*2 but that was getting too many posts
                time_filter='year'  # tried 'month' first but not enough data
            ))
            
            # shuffle to get random sample
            random.shuffle(search_results)
            
            post_count = 0
            for post in search_results:
                if total_posts_collected >= limit:  # double check the limit
                    break
                    
                post_date = datetime.fromtimestamp(post.created_utc)
                if start_date <= post_date <= end_date:
                    posts_list.append({
                        'topic': topic,
                        'title': post.title,
                        'text': post.selftext,
                        'score': post.score,
                        'created_utc': post_date,
                        'subreddit': subreddit_name,
                        'analysis_type': analysis_type
                    })
                    post_count += 1
                    total_posts_collected += 1
            
            if post_count > 0:
                successful_attempts += 1
                
        except Exception as e:
            # just skip if there's an error
            continue
    
    if successful_attempts == 0:
        st.warning(f"No data found for {topic}. Tried {total_attempts} subreddits but all failed.")
    else:
        st.success(f"Successfully collected {len(posts_list)} posts for {topic} from {successful_attempts} subreddits.")
    
    # make sure we don't return more than requested
    return posts_list[:limit]

# Clean up the text data
def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    # remove URLs - they're not useful for sentiment
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    # remove punctuation and special chars
    text = re.sub(r'[^\w\s]', '', text)
    # clean up extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text

    # Get sentiment score using VADER
    @st.cache_data
    def get_sentiment(text):
        analyzer = SentimentIntensityAnalyzer()
        scores = analyzer.polarity_scores(text)
        return scores['compound']  # compound score worked best after testing different options

# Convert numeric score to category
def categorize_sentiment(score):
    if score >= 0.05:
        return 'Positive'
    elif score <= -0.05:
        return 'Negative'
    else:
        return 'Neutral'

# Create the sentiment distribution chart
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

# Box plot for sentiment scores
def plot_sentiment_scores(df, analysis_type):
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(x='topic', y='sentiment_score', data=df, ax=ax)
    plt.title(f'Sentiment Score Distribution by {analysis_type}')
    plt.xlabel(analysis_type)
    plt.ylabel('Sentiment Score')
    plt.xticks(rotation=45)
    plt.tight_layout()
    return fig

# Monthly trends over time
def plot_monthly_trends(df, topics, analysis_type):
    # Convert to monthly periods
    df['month'] = df['created_utc'].dt.to_period('M')
    
    # Get monthly averages
    monthly_sentiment = df.groupby(['month', 'topic'])['sentiment_score'].mean().unstack()
    
    fig, ax = plt.subplots(figsize=(15, 8))
    
    for topic in topics:
        if topic in monthly_sentiment.columns:
            plt.plot(range(len(monthly_sentiment.index)), 
                    monthly_sentiment[topic], 
                    label=topic, 
                    marker='o')
    
    # Set month labels on x-axis
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

# Weekly heatmap - this one was tricky to get right
def plot_weekly_heatmap(df, analysis_type):
    fig, ax = plt.subplots(figsize=(12, 8))
    weekly_sentiment = df.groupby(['week', 'topic'])['sentiment_score'].mean().unstack()
    sns.heatmap(weekly_sentiment, cmap='RdYlGn', center=0, annot=True, fmt='.2f', ax=ax)
    plt.title(f'Weekly Sentiment Patterns by {analysis_type}')
    plt.xlabel(analysis_type)
    plt.ylabel('Week Number')
    plt.tight_layout()
    return fig

# Main app function
def main():
    st.title('Cinema Analysis')
    st.write('Analyzing Reddit sentiment about streaming platforms, directors, and international cinema')

    # Sidebar for controls
    st.sidebar.header('Analysis Controls')
    
    # Choose what type of analysis to run
    analysis_type = st.sidebar.radio(
        'Select Analysis Type',
        ['Streaming Platforms', 'Directors', 'International Cinema']
    )
    
    # Different options based on analysis type
    if analysis_type == 'Streaming Platforms':
        platform_options = [
            'Netflix', 'Hulu', 'Disney+', 'HBO Max', 'Prime Video',
            'Apple TV+', 'Paramount+', 'Peacock', 'Tubi', 'Criterion Channel',
            'MUBI', 'Shudder', 'Crackle', 'Pluto TV', 'YouTube Premium'
        ]
        selected_topics = st.sidebar.multiselect(
            'Select Platforms to Analyze',
            platform_options,
            default=['Netflix', 'Hulu', 'Disney+']
        )
        display_type = "Platform"
    elif analysis_type == 'Directors':
        director_list = [
            'Bernardo Bertolucci', 'Martin Scorsese', 'Satyajit Ray', 'Jean-Luc Godard',
            'Andrei Tarkovsky', 'Wim Wenders', 'Denis Villeneuve', 'Abbas Kiarostami',
            'Majid Majidi', 'Terrence Malick', 'Greta Gerwig', 'Christopher Nolan',
            'Michael Mann', 'Wong Kar-wai', 'Akira Kurosawa', 'Ingmar Bergman',
            'Federico Fellini', 'Alfred Hitchcock', 'Stanley Kubrick', 'David Lynch',
            'Hayao Miyazaki', 'Park Chan-wook', 'Bong Joon-ho', 'Pedro Almodóvar',
            'Lars von Trier', 'David Fincher', 'Quentin Tarantino', 'Coen Brothers',
            'Wes Anderson', 'Guillermo del Toro'
        ]
        
        # Let users add custom directors
        custom_director_input = st.sidebar.text_input('Or enter a custom director name:').strip()
        if custom_director_input and custom_director_input not in director_list and len(custom_director_input) > 2:
            # Add to the list
            director_list = sorted(director_list + [custom_director_input])
            st.sidebar.success(f"Added {custom_director_input} to the list!")
        
        selected_topics = st.sidebar.multiselect(
            'Select Directors to Analyze',
            director_list,
            default=['Martin Scorsese', 'Christopher Nolan', 'Denis Villeneuve']
        )
        display_type = "Director"
    else:  # International Cinema
        country_list = [
            'France', 'Italy', 'Japan', 'South Korea', 'India', 'Iran',
            'Germany', 'Sweden', 'Russia', 'Spain', 'Mexico', 'Brazil',
            'China', 'Hong Kong', 'Taiwan', 'Thailand', 'Turkey', 'Egypt',
            'Argentina', 'Denmark', 'Poland', 'Czech Republic', 'Hungary',
            'Romania', 'Greece', 'Belgium', 'Netherlands', 'Norway',
            'Finland', 'Austria', 'Vietnam', 'Philippines', 'Indonesia',
            'Chile', 'Colombia', 'Venezuela', 'Morocco', 'Nigeria',
            'South Africa', 'Israel', 'Lebanon'
        ]
        
        # Custom country input
        custom_country_input = st.sidebar.text_input('Or enter a custom country name:').strip()
        if custom_country_input and custom_country_input not in country_list and len(custom_country_input) > 2:
            # Add to list
            country_list = sorted(country_list + [custom_country_input])
            st.sidebar.success(f"Added {custom_country_input} to the list!")
        
        selected_topics = st.sidebar.multiselect(
            'Select Countries to Analyze',
            country_list,
            default=['France', 'Japan', 'South Korea']
        )
        display_type = "Country"

    # How many posts to analyze per topic
    posts_limit = st.sidebar.slider(
        'Posts per Topic',
        min_value=10,
        max_value=200,
        value=100,
        step=10
    )

    # Try to connect to Reddit
    try:
        reddit = init_reddit()
        
        if st.button('Run Analysis'):
            with st.spinner('Collecting and analyzing Reddit posts...'):
                # Collect data for all selected topics
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
                
                # Create DataFrame from collected posts
                df = pd.DataFrame(all_posts)
                
                if len(df) > 0:
                    # Process the data
                    df['clean_title'] = df['title'].apply(clean_text)
                    df['clean_text'] = df['text'].apply(clean_text)
                    df['combined_text'] = df['clean_title'] + ' ' + df['clean_text']
                    df['sentiment_score'] = df['combined_text'].apply(get_sentiment)
                    df['sentiment'] = df['sentiment_score'].apply(categorize_sentiment)
                    df['date'] = df['created_utc'].dt.date
                    df['week'] = df['created_utc'].dt.isocalendar().week

                    # Show results in different tabs
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
                        
                        # Show overall sentiment breakdown
                        st.subheader("Overall Sentiment Distribution")
                        sentiment_counts = df['sentiment'].value_counts(normalize=True).round(3)
                        col1, col2, col3 = st.columns(3)
                        col1.metric("Positive", f"{sentiment_counts.get('Positive', 0):.1%}")
                        col2.metric("Neutral", f"{sentiment_counts.get('Neutral', 0):.1%}")
                        col3.metric("Negative", f"{sentiment_counts.get('Negative', 0):.1%}")

                        # Posts per topic chart
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

                    # Option to show raw data
                    if st.checkbox('Show Raw Data'):
                        st.subheader('Raw Data')
                        st.dataframe(df[['topic', 'title', 'sentiment', 'sentiment_score', 'created_utc', 'subreddit']])
                
                else:
                    st.error("No posts found for the selected topics.")
                
    except Exception as e:
        st.error(f"Error connecting to Reddit API: {str(e)}")

if __name__ == "__main__":
    main() 