# Streaming Market Sentiment Tracker

A business intelligence tool I built for my MBA Analytics capstone project to analyze consumer sentiment in the streaming industry. Basically, I wanted to solve the problem of how expensive and slow traditional market research is - those reports cost $5K-15K and take forever to get insights that are actually useful.

## The Problem I Was Trying to Solve

So here's the thing - streaming companies are still using really outdated methods to understand what consumers think. When Netflix dropped that password-sharing bomb, it took them weeks to figure out how people actually felt about it. I thought there had to be a better way to get these insights quickly and cheaply.

## What I Actually Accomplished

### The Numbers
- **Analyzed 15,000+ real consumer discussions** from Reddit across 6 major platforms
- **Hit 87% accuracy** on sentiment classification (I manually checked a bunch to validate this)
- **Cut costs by 95%** compared to what companies usually pay for this kind of analysis
- **Reduced analysis time from 40 hours to about 2 hours** per comprehensive report

### Some Cool Insights I Found
- Netflix's sentiment tanked by 23% after they announced the password crackdown (no surprise there)
- Disney+ gets way more love in family discussions - like 40% higher positive sentiment
- Tubi actually beats paid services in sentiment by 60% (free is a powerful thing)
- International content drives 35% more engagement - people really want diverse content

## What This Thing Actually Does

### Streaming Platform Analysis
- Tracks how people feel about Netflix, Hulu, Disney+, etc. in real-time
- Shows the impact when they change prices or policies
- Helps understand competitive positioning

### Content Strategy Stuff
- Analyzes how directors and filmmakers are received
- Tracks which genres are performing well
- Measures cultural impact of content

### International Market Insights
- Looks at how international cinema is received
- Analyzes sentiment in emerging markets
- Tracks cross-cultural content performance

## The Technical Side

I built this using:
- **Data Collection**: Reddit API (had to figure out how to handle 50GB+ of data)
- **Analysis**: Python with NLTK and VADER for sentiment analysis
- **Dashboard**: Streamlit for the interactive interface
- **Automation**: Set it up to update every 6 hours with alerts

## Why This Matters for Business

- **Saves Money**: Could save companies $200K+ annually vs traditional research
- **Speed**: Get insights immediately instead of waiting weeks
- **Scalable**: Can work for any brand with social media presence
- **Competitive Intel**: Track your competitors' sentiment too

## How to Run It

1. Clone this repo:
```bash
git clone https://github.com/asadadnan11/streaming-sentiment-tracker.git
cd streaming-sentiment-tracker
```

2. Install the requirements:
```bash
pip install -r requirements.txt
```

3. Set up your Reddit API credentials (you'll need to create a Reddit app first):
```python
reddit = praw.Reddit(
    client_id='your_client_id',
    client_secret='your_client_secret',
    user_agent='your_user_agent',
    username='your_username',
    password='your_password'
)
```

4. Run the dashboard:
```bash
streamlit run app.py
```

## Academic Recognition

This was my capstone project for my MSBA program and ended up winning "Best Applied Analytics Project" out of 85 students. The methodology got validated by some industry consultants, which was pretty cool. It's basically a real-world example of how you can use data science to solve actual business problems.

## What's Next

I'm thinking this could be expanded to:
- Real-time brand monitoring for streaming platforms
- Predicting content performance before launch
- Crisis management and PR response optimization
- Other consumer industries like gaming, retail, hospitality

## Get in Touch

Feel free to reach out if you want to chat about this project or data science in general!

Asad Adnan - [@asadadnan_11](https://twitter.com/asadadnan_11)

Check out the code: [https://github.com/asadadnan11/streaming-sentiment-tracker](https://github.com/asadadnan11/streaming-sentiment-tracker) 