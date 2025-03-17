# News-Article-Recommandation
Used NLP to fetch news from the online based on the user input 
News Article Recommender
This Python script fetches and recommends news articles based on user input using web scraping and NLP techniques.

Features
Fetches news articles from Google News based on user queries.
Uses NLTK for text preprocessing (stopword removal, tokenization, and lemmatization).
Implements TF-IDF Vectorization to compute cosine similarity for recommendation.
Recommends the most relevant news articles based on user queries.
Requirements
Before running the script, install the required dependencies:

bash
Copy
Edit
pip install requests beautifulsoup4 nltk scikit-learn
Setup
Download necessary NLTK datasets:
python
Copy
Edit
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
Run the script:
bash
Copy
Edit
python news_recommender.py
How It Works
The script asks for a topic of interest.
It scrapes Google News for relevant articles.
It preprocesses the text (tokenization, stopword removal, and lemmatization).
It computes similarity scores using TF-IDF and recommends the most relevant news articles.
Limitations
Google may block automated scraping. Consider using NewsAPI or GNews API for reliable results.
The scraping structure may change, requiring updates to the HTML parsing logic.
Future Enhancements
Integrate with NewsAPI or another news provider for more stability.
Improve text analysis with advanced NLP models.
Implement a Flask web app for user-friendly interaction.

