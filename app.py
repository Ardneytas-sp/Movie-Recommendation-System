from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from tensorflow.keras import layers
import tensorflow as tf
from pathlib import Path
import os
import tmdbsimple

# Set your TMDb API key
tmdb_api_key = '037b4ab3d95b71d0de81bb0' # change with your API , it's a pseudo key 

tmdbsimple.API_KEY = tmdb_api_key

app = Flask(__name__)
movielens_data_file_url = (
    "http://files.grouplens.org/datasets/movielens/ml-latest-small.zip"
)
movielens_zipped_file = tf.keras.utils.get_file(
    "ml-latest-small.zip", movielens_data_file_url, extract=False
)
keras_datasets_path = Path(movielens_zipped_file).parents[0]
movielens_dir = keras_datasets_path / "ml-latest-small"
EMBEDDING_SIZE = 50
# Load your data
ratings_file = movielens_dir / "ratings.csv"
df = pd.read_csv(ratings_file)
movie_df = pd.read_csv(movielens_dir / 'movies.csv')

# Map user ID to a "user vector" via an embedding matrix
user_ids = df["userId"].unique().tolist()
num_users = len(user_ids)

# Map movies ID to a "movies vector" via an embedding matrix
movie_ids = df["movieId"].unique().tolist()
num_movies = len(movie_ids)

# Collaborative filtering variables
user2user_encoded, user_encoded2user = {}, {}
for i, user_id in enumerate(user_ids):
    user2user_encoded[user_id] = i
    user_encoded2user[i] = user_id

movie2movie_encoded, movie_encoded2movie = {}, {}
for i, movie_id in enumerate(movie_ids):
    movie2movie_encoded[movie_id] = i
    movie_encoded2movie[i] = movie_id


class RecommenderNet(tf.keras.Model):
    def __init__(self, num_users, num_movies, embedding_size, **kwargs):
        super(RecommenderNet, self).__init__(**kwargs)
        self.num_users = num_users
        self.num_movies = num_movies
        self.embedding_size = embedding_size
        self.user_embedding = layers.Embedding(
            num_users,
            embedding_size,
            embeddings_initializer="he_normal",
            embeddings_regularizer=tf.keras.regularizers.l2(1e-6),
        )
        self.user_bias = layers.Embedding(num_users, 1)
        self.movie_embedding = layers.Embedding(
            num_movies,
            embedding_size,
            embeddings_initializer="he_normal",
            embeddings_regularizer=tf.keras.regularizers.l2(1e-6)
        )
        self.movie_bias = layers.Embedding(num_movies, 1)

    def call(self, inputs):
        user_vector = self.user_embedding(inputs[:, 0])
        user_bias = self.user_bias(inputs[:, 0])
        movie_vector = self.movie_embedding(inputs[:, 1])
        movie_bias = self.movie_bias(inputs[:, 1])
        dot_user_movie = tf.tensordot(user_vector, movie_vector, 2)
        # Add all the components (including bias)
        x = dot_user_movie + user_bias + movie_bias
        # The sigmoid activation forces the rating to be between 0 and 11
        return tf.nn.sigmoid(x)


model = RecommenderNet(num_users, num_movies, EMBEDDING_SIZE)

file_path = os.path.join("C:\\Users\Satyendra Singh\Documents\movie_recommendation_files", "collaborativeFilteringmodel_weights.h5")
model.build((None, 2))

# Load the model weights
model.load_weights(file_path, by_name=True)

# Function to get movie recommendations
def get_movie_recommendations(user_id, model, movie_df, df, user2user_encoded, movie2movie_encoded, movie_encoded2movie):
    # Get movies watched by the user
    movies_watched_by_user = df[df.userId == user_id]

    # Get movies not watched by the user
    movies_not_watched = movie_df[~movie_df['movieId'].isin(movies_watched_by_user.movieId.values)]['movieId']
    movies_not_watched = list(set(movies_not_watched).intersection(set(movie2movie_encoded.keys())))
    movies_not_watched = [[movie2movie_encoded.get(x)] for x in movies_not_watched]

    # Encode the user
    user_encoder = user2user_encoded.get(user_id)

    # Create input array for the model
    user_movie_array = np.hstack(
        ([[user_encoder]] * len(movies_not_watched), movies_not_watched)
    )

    # Predict ratings using the model
    ratings = model.predict(user_movie_array).flatten()

    # Get top-rated movie indices
    top_ratings_indices = ratings.argsort()[-10:][::-1]

    # Get recommended movie IDs
    recommended_movie_ids = [movie_encoded2movie.get(movies_not_watched[x][0]) for x in top_ratings_indices]

    # Get movie titles and posters from TMDb API
    recommended_movies = []
    recommended_posters = []
    for movie_id in recommended_movie_ids:
        # Fetch movie details from TMDb
        search = tmdbsimple.Movies(movie_id)
        try:
            movie_info = search.info()
            recommended_movie_title = movie_info['title']
            recommended_movies.append(recommended_movie_title)
            recommended_posters.append(f"https://image.tmdb.org/t/p/w500{movie_info['poster_path']}")
        except:
            print(f"Movie ID {movie_id} not found on TMDb.")

    return recommended_movies, recommended_posters

# Flask routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/recommendations', methods=['POST'])
def recommendations():
    user_id = int(request.form['user_id'])
    recommendations, posters = get_movie_recommendations(
        user_id=user_id,
        model=model,
        movie_df=movie_df,
        df=df,  # Assuming df is your movie dataframe
        user2user_encoded=user2user_encoded,
        movie2movie_encoded=movie2movie_encoded,
        movie_encoded2movie=movie_encoded2movie
    )
    return render_template('recommendation.html', user_id=user_id, recommendations=recommendations, posters=posters)

 
if __name__ == '__main__':
    app.run(debug=True)
