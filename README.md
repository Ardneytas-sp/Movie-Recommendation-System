🎬 Movie Recommendation System

📌 Overview

The Movie Recommendation System is a machine learning-based application that suggests movies to users based on their viewing history or the similarity of movies they like. This project demonstrates the implementation of Content-Based Filtering (and/or Collaborative Filtering) algorithms to provide personalized recommendations.

🚀 Features

Search Functionality: Users can search for a movie from the database.

Personalized Recommendations: Returns a list of top 7-10 similar movies based on the user's selection.

Data Visualization: Includes insights on the dataset (e.g., most popular genres, top-rated movies).

Interactive UI:  Built with Streamlit/Flask for an easy-to-use web interface.


🛠️ Tech Stack

Language: Python
Libraries:
  Pandas (Data Manipulation).

  NumPy (Numerical Computations).

  Scikit-learn (Cosine Similarity, CountVectorizers).

  NLTK (Natural Language Processing -  identifying keywords/tags).

  Streamlit (For the Web App interface).


📊 How It Works

1. Data Preprocessing: The system cleans the dataset, handling missing values and merging relevant columns (Overview, Cast, Crew, Keywords).

2. Feature Extraction: Text data is converted into vectors using Bag of Words (CountVectorizer) or TF-IDF.

3. Similarity Calculation: We calculate the Cosine Similarity between vectors to find movies that are closest to each other in vector space.

4. Recommendation: When a user selects a movie, the system retrieves and displays the movies with the highest similarity scores
