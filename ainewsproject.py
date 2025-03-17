import requests
from bs4 import BeautifulSoup
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Download required NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Function to fetch news from Google News based on user query
def fetch_news(query):
    search_url = f"https://www.google.com/search?q={query.replace(' ', '+')}&tbm=nws"
    response = requests.get(search_url)
    
    if response.status_code != 200:
        return None
    
    soup = BeautifulSoup(response.content, 'html.parser')
    articles = []
    
    # Scraping news titles and URLs from Google News search results
    for g in soup.find_all('div', class_='BNeawe vvjwJb AP7Wnd'):
        title = g.get_text()
        parent = g.find_parent('a')
        link = parent['href']
        full_link = "https://www.google.com" + link
        articles.append((title, full_link))
    
    return articles

# Function to preprocess text
def preprocess_text(text):
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    
    tokens = word_tokenize(text.lower())
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words and word not in string.punctuation]
    
    return ' '.join(tokens)

# Function to recommend articles based on user query
def recommend_articles(user_query, article_summaries):
    user_query_processed = preprocess_text(user_query)
    
    # Create a list of documents (user query + article summaries)
    documents = [user_query_processed] + article_summaries
    
    # Vectorize the documents
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(documents)
    
    # Compute cosine similarity between the user query and each article
    cosine_similarities = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:])
    
    # Sort articles based on similarity scores
    similarity_scores = list(enumerate(cosine_similarities[0]))
    similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)
    
    # Return sorted indices of the top matching articles
    sorted_indices = [idx for idx, score in similarity_scores]
    
    return sorted_indices

# Main program to take user input and provide news recommendations
def main():
    # User directly enters any topic of interest
    user_query = input("What type of news article do you want? ")
    
    # Fetch news based on user query
    articles = fetch_news(user_query)
    
    if not articles:
        print("Sorry, no relevant news found for your query. Please try another topic.")
        return

    # Preprocess the news titles for recommendation
    article_summaries = [preprocess_text(article[0]) for article in articles]
    
    # Get recommended article indices based on user query
    recommended_indices = recommend_articles(user_query, article_summaries)
    
    # Display top recommendations with links
    print("\nRecommended News Articles:")
    for idx in recommended_indices[:5]:  # Show top 5 recommendations
        title, link = articles[idx]
        print(f"Title: {title}")
        print(f"Link: {link}")
        print()

# Run the main program
if __name__ == "__main__":
    main()
