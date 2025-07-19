from pathlib import Path
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix, save_npz,load_npz
from sklearn.metrics.pairwise import cosine_similarity




def Normalize(content_scores):
    # Normalize content scores (scale to [0, 1])
    content_min = np.min(content_scores)
    content_max = np.max(content_scores)
    if content_max - content_min != 0:
        content_norm = (content_scores - content_min) / (content_max - content_min)
        return content_norm
    else:
        content_norm = content_scores 
        return content_norm


def hybrid_rec(song_name, artist_name,songs_data, collaborative_matrix, content_matrix,k,content_based_weight,collaborative_weight):
 # Find the song in the dataset
    song_row = songs_data[
        (songs_data['name'].str.lower() == song_name.lower()) & 
        (songs_data['artist'].str.lower() == artist_name.lower())
    ]
    
    if song_row.empty:
        raise ValueError(f"Song '{song_name}' by '{artist_name}' not found.")
    
    songs_data['track_id'] = songs_data['track_id'].astype('category')

    # Get track_id and index
    input_track_id = song_row['track_id'].values[0]
    print(input_track_id)
    input_track_index = np.where(songs_data['track_id'].cat.categories == input_track_id)[0][0]
    print(input_track_index)

    input_array_collaborative = collaborative_matrix[input_track_index]
    input_array_content = content_matrix[input_track_index]

    # Get similarity scores from both matrices
    collaborative_scores = cosine_similarity(input_array_collaborative, collaborative_matrix)
    content_scores = cosine_similarity(input_array_content, content_matrix)

    norm_collaborative_scores = Normalize(collaborative_scores)
    norm_content_scores =  Normalize(content_scores)


    wieghted_scores = collaborative_weight * norm_collaborative_scores + content_based_weight * norm_content_scores
    print("weighted scores : ",wieghted_scores)
    print("wieghted argument",np.argsort(-wieghted_scores))

    # Get top k recommendations
    top_k_indices = np.argsort(-wieghted_scores).ravel()[:k+1]

    return(
        songs_data
        .iloc[top_k_indices]
        .reset_index(drop=True)
    )
    



def main():
    current_dir = Path().cwd()

    # Load data
    collaborative_matrix = load_npz(current_dir / "data" / "item_user_matrix.npz")
    content_matrix = load_npz(current_dir / "data" / "content_based.npz")
    songs_data = pd.read_csv(current_dir / "data" / "filtered_songs.csv")

    # Prepare songs data
    songs_data["track_id"] = songs_data["track_id"].astype("category")

    # Get recommendations
    # song_name = "Shake It Off"
    # artist_name = 'Taylor Swift'
    
    
    # recommendations = hybrid_rec(
    #         song_name=song_name,
    #         artist_name=artist_name,
    #         collaborative_matrix=collaborative_matrix,  
    #         content_matrix=content_matrix,
    #         songs_data=songs_data,
    #         utility_df=user_listened_data  
    #     )


if __name__ == "__main__":
    main()