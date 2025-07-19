import numpy as np
import pandas as pd
import joblib
import sys
from sklearn.preprocessing import MinMaxScaler, StandardScaler, OneHotEncoder
from category_encoders.count import CountEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.compose import ColumnTransformer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import save_npz, csr_matrix
from pathlib import Path

def train_transformer(data, frequency_encode_cols, ohe_cols, tfidf_col, standard_scale_cols, min_max_scale_cols):
    """
    Trains and saves a ColumnTransformer for data preprocessing.
    """
    transformer = ColumnTransformer(
        transformers=[
            ("frequency_encode", CountEncoder(normalize=True, return_df=True), frequency_encode_cols),
            ("ohe", OneHotEncoder(handle_unknown="ignore"), ohe_cols),
            ("tfidf", TfidfVectorizer(max_features=85), tfidf_col),
            ("standard_scale", StandardScaler(), standard_scale_cols),
            ("min_max_scale", MinMaxScaler(), min_max_scale_cols)
        ],
        remainder='passthrough',
        n_jobs=-1
    )

    transformer.fit(data)
    joblib.dump(transformer, "transformer.joblib")

def data_for_content_filtering(data: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans the input DataFrame by dropping specific columns.
    This function removes the columns "track_id", "name", and "spotify_preview_url"
    from the provided DataFrame.

    Parameters:
    data (pd.DataFrame): The input DataFrame containing music track information.

    Returns:
    pd.DataFrame: A DataFrame with the specified columns removed.
    """
    filtered_data = data.drop(columns=["track_id", "name", "spotify_preview_url"], errors="ignore")
    return filtered_data

def transform_data(data):
    """
    Loads the trained transformer and applies transformations to new data.
    """
    transformer = joblib.load("transformer.joblib")
    transformed_data = transformer.transform(data)

    if not isinstance(transformed_data, csr_matrix):
        transformed_data = csr_matrix(transformed_data)
    
    return transformed_data

def save_transformed_data(transformed_data, save_path):
    """
    Saves transformed data as a sparse matrix.
    """
    save_npz(save_path, transformed_data)

def recommend(song_name, artist_name, songs_data, transformed_data, k=10):
    """
    Recommends top k songs similar to the given song.
    """
    song_name = song_name.strip().lower()
    song_row = songs_data[(songs_data["name"].str.lower() == song_name.lower()) & (songs_data["artist"].str.lower() == artist_name.lower())]

    if song_row.empty:
        return None

    song_index = song_row.index[0]
    input_vector = transformed_data[song_index].reshape(1, -1)
    similarity_scores = cosine_similarity(input_vector, transformed_data)
    top_k_songs_indexes = np.argsort(similarity_scores.ravel())[-k:][::-1]
    top_k_songs = songs_data.iloc[top_k_songs_indexes]
    
    return top_k_songs[['name', 'artist', 'spotify_preview_url']].reset_index(drop=True)

def main():
    """
    Main function to run the content filtering pipeline.
    """
    cleaned_data = pd.read_csv("data/cleaned_data.csv")
    df_content_similarity = data_for_content_filtering(cleaned_data)

    train_transformer(
        df_content_similarity,
        frequency_encode_cols=["year"],
        ohe_cols=["artist", "time_signature", "key"],
        tfidf_col="tags",
        standard_scale_cols=["duration_ms", "loudness", "tempo"],
        min_max_scale_cols=["danceability", "energy", "speechiness", "acousticness", "instrumentalness", "liveness", "valence"]
    )

    transformed_data = transform_data(df_content_similarity)
    save_transformed_data(transformed_data, "data/transformed_data.npz")

if __name__ == "__main__":
    main()