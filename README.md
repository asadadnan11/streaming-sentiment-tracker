# 🎬 Streaming & Cinema Sentiment Tracker

I built this tool to analyze how people feel about different aspects of cinema - from streaming platforms to legendary directors. It scrapes Reddit discussions and uses sentiment analysis to track public opinion.

## What It Does

### Streaming Platforms 📺
Track how people feel about:
- Netflix, Hulu, Disney+ and other platforms
- Price changes and new content drops
- Service quality and user experience

### Film Directors 🎥
Analyze discussions about your favorite directors:
- Modern masters like Nolan, Villeneuve, Fincher
- Legends like Kubrick, Kurosawa, Tarkovsky
- Add any director you're interested in

### International Cinema 🌏
See how different national cinemas are received:
- French, Japanese, Korean films
- Emerging film industries
- Global cinema trends

## Tech Stack

- **Data**: Reddit API (PRAW)
- **Analysis**: Python, NLTK for sentiment analysis
- **Interface**: Streamlit for a clean, interactive experience
- **Visualization**: Matplotlib & Seaborn

## Quick Start

1. Clone it:
```bash
git clone https://github.com/asadadnan11/streaming-sentiment-tracker.git
cd streaming-sentiment-tracker
```

2. Install what you need:
```bash
pip install -r requirements.txt
```

3. Set up Reddit API (you'll need your own credentials):
```python
reddit = praw.Reddit(
    client_id='your_client_id',
    client_secret='your_client_secret',
    user_agent='your_user_agent',
    username='your_username',
    password='your_password'
)
```

4. Fire it up:
```bash
streamlit run app.py
```

## What You Can Do

- Compare sentiment across different streaming platforms
- Track how opinions change over time
- Analyze discussions about specific directors
- Monitor international cinema trends
- Generate visual insights about cinema discussions

## Contact

Asad Adnan - [@asadadnan_11](https://twitter.com/asadadnan_11)

Check it out: [https://github.com/asadadnan11/streaming-sentiment-tracker](https://github.com/asadadnan11/streaming-sentiment-tracker) 