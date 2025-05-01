# 🎬 Cinema Sentiment Analysis

A powerful sentiment analysis tool that explores public opinion about streaming platforms, film directors, and international cinema using Reddit data. Built with Python, NLTK, and Streamlit.

![Cinema Analysis Banner](https://i.imgur.com/your_banner_image.jpg)

## 🌟 Features

### Multi-Domain Analysis
- **Streaming Platforms** 📺
  - Track sentiment for Netflix, Hulu, Disney+, and more
  - Analyze weekly and monthly trends
  - Monitor reactions to price changes and new content

- **Film Directors** 🎥
  - Analyze sentiment for legendary and contemporary directors
  - Compare reception across different directorial styles
  - Add and analyze custom director entries

- **International Cinema** 🌏
  - Track sentiment towards different national film industries
  - Monitor monthly trends in global cinema reception
  - Analyze emerging film markets

### Advanced Analytics
- Sentiment scoring using VADER analysis
- Temporal trend analysis
- Distribution visualization
- Cross-platform comparisons

### Interactive Visualizations
- Sentiment distribution charts
- Time series analysis
- Weekly pattern heatmaps
- Comparative box plots

## 🚀 Getting Started

### Prerequisites
- Python 3.8+
- Reddit API credentials

### Installation

1. Clone the repository:
```bash
git clone https://github.com/asadadnan11/cinema-sentiment-analysis.git
cd cinema-sentiment-analysis
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up your Reddit API credentials:
```python
reddit = praw.Reddit(
    client_id='your_client_id',
    client_secret='your_client_secret',
    user_agent='your_user_agent',
    username='your_username',
    password='your_password'
)
```

4. Run the Streamlit app:
```bash
streamlit run app.py
```

## 📊 Usage Examples

### Analyzing Streaming Platforms
```python
# Select platforms to analyze
platforms = ['Netflix', 'Disney+', 'Hulu']
# Set post limit
posts_limit = 100
# Run analysis
```

### Director Analysis
```python
# Compare directors
directors = ['Christopher Nolan', 'Martin Scorsese']
# Analyze sentiment
```

### International Cinema
```python
# Select countries
countries = ['South Korea', 'France', 'Japan']
# Track monthly trends
```

## 📈 Sample Visualizations

### Sentiment Distribution
![Sentiment Distribution](https://i.imgur.com/your_sentiment_dist.jpg)
- Compare positive/neutral/negative sentiment across categories

### Time Trends
![Time Trends](https://i.imgur.com/your_time_trends.jpg)
- Track sentiment changes over time

### Pattern Analysis
![Pattern Analysis](https://i.imgur.com/your_pattern_analysis.jpg)
- Analyze weekly and monthly patterns

## 🛠️ Technical Details

### Data Collection
- Reddit API integration via PRAW
- Multi-subreddit search
- Smart post filtering and validation

### Sentiment Analysis
- NLTK VADER sentiment analyzer
- Compound score calculation
- Categorical sentiment classification

### Visualization Stack
- Streamlit for interactive UI
- Matplotlib for static plots
- Seaborn for statistical visualizations

## 🎯 Use Cases

1. **Industry Analysis**
   - Track streaming platform reception
   - Monitor content strategy impact
   - Analyze market trends

2. **Film Studies**
   - Compare directorial styles
   - Analyze cultural reception
   - Track artistic influence

3. **Market Research**
   - Monitor audience preferences
   - Track emerging trends
   - Analyze competitive landscape

## 🤝 Contributing

Contributions are welcome! Here's how you can help:

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- NLTK team for VADER sentiment analysis
- Reddit API for data access
- Streamlit team for the amazing framework

## 📬 Contact

Asad Adnan - [@asadadnan_11](https://twitter.com/asadadnan_11)

Project Link: [https://github.com/asadadnan11/cinema-sentiment-analysis](https://github.com/asadadnan11/cinema-sentiment-analysis)

---

<p align="center">
Made with ❤️ for cinema enthusiasts and data lovers
</p> 