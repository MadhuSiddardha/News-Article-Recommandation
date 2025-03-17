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

# Function to scrape Wikipedia page
def fetch_wikipedia_content(query):
    url = f"https://en.wikipedia.org/wiki/{query.replace(' ', '_')}"
    response = requests.get(url)
    
    if response.status_code != 200:
        return None, None  # Return None if the page isn't found
    
    soup = BeautifulSoup(response.content, 'html.parser')
    
    content = ''
    for paragraph in soup.find_all('p'):
        content += paragraph.get_text()
    
    return content, url

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

# Main program to take user input and provide recommendations
def main():
    # Expanded input prompt with more topics for the user to choose from
    user_query = input("Enter a topic you are interested in (e.g., 'politics', 'technology', 'sports', 'crimes', 'environment', 'health', 'education', 'finance'): ")
    
    # Fetch content from Wikipedia based on the query
    content, url = fetch_wikipedia_content(user_query)
    
    if content is None:
        print("Sorry, no information available for that topic. Please try another.")
        return

    # Preprocess the fetched content
    processed_content = preprocess_text(content)
    
    # Example list of pre-fetched article summaries
    # Expanded list with new topics: crimes and environment
    article_summaries = [
        preprocess_text("Artificial intelligence is intelligence demonstrated by machines, in contrast to the natural intelligence displayed by humans."),
        preprocess_text("Technology is the sum of techniques, skills, methods, and processes used in the production of goods or services."),
        preprocess_text("Sports are all forms of competitive physical activity or games which, through casual or organized participation, aim to use, maintain or improve physical ability and skills while providing enjoyment to participants."),
        preprocess_text("Crime is an unlawful act punishable by a state or other authority. The term 'crime' does not, in modern criminal law, have any simple and universally accepted definition."),
        preprocess_text("Environmental issues are harmful effects of human activity on the biophysical environment. They include pollution, climate change, and depletion of resources."),
        preprocess_text("Health is a state of physical, mental and social well-being in which disease and infirmity are absent."),
        preprocess_text("Education is the process of facilitating learning, or the acquisition of knowledge, skills, values, morals, beliefs, and habits."),
        preprocess_text("Finance is the study and discipline of money, currency, and capital assets. It is related to economics, the study of production, distribution, and consumption of goods and services."),
        processed_content  # Adding the fetched content from Wikipedia
    ]

    article_links = [
        "https://en.wikipedia.org/wiki/Artificial_intelligence",
        "https://en.wikipedia.org/wiki/Technology",
        "https://en.wikipedia.org/wiki/Sport",
        "https://en.wikipedia.org/wiki/Crime",
        "https://en.wikipedia.org/wiki/Environmental_issues",
        "https://en.wikipedia.org/wiki/Health",
        "https://en.wikipedia.org/wiki/Education",
        "https://en.wikipedia.org/wiki/Finance",
        url  # Adding the link for fetched content
    ]
    
    # Get recommended article indices based on user query
    recommended_indices = recommend_articles(user_query, article_summaries)
    
    # Display recommendations with links
    print("\nRecommended Articles:")
    for idx in recommended_indices[:3]:  # Show top 3 recommendations
        print(f"Title: {article_summaries[idx][:50]}...")  # Display first 50 characters of the summary as title
        print(f"Link: {article_links[idx]}")
        print()

# Run the main program
if __name__ == "__main__":
    main()
