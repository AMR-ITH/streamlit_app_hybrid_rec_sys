from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.sparse import csr_matrix, save_npz,load_npz
from sklearn.metrics.pairwise import cosine_similarity

def save_user_listening_data(cwd_path,cleaned_songs: pd.DataFrame, user_history: pd.DataFrame) -> None:
    """
    This function reads the cleaned songs data and the user listening history data
    and returns the user listening data in a DataFrame.

    Parameters:
    cleaned_songs_path (Path): The path to the cleaned songs data.
    user_history_path (Path): The path to the user listening history data.        

    Returns:
    pd.DataFrame: The user listening data.
    """


    unique_track_ids = user_history["track_id"].unique()

    # Create a copy of the filtered DataFrame to avoid SettingWithCopyWarning
    filtered_songs = cleaned_songs[cleaned_songs["track_id"].isin(unique_track_ids)].copy()
    filtered_songs.loc[:, 'track_id'] = filtered_songs['track_id'].astype('category')
    filtered_songs.sort_values("track_id", inplace=True)
    filtered_songs.reset_index(drop=True, inplace=True)

    print(f"Filtered songs data shape: {filtered_songs.shape}")

    path_loc = cwd_path / "data" / "filtered_songs.csv"
    filtered_songs.to_csv(path_loc, index=False)

    return filtered_songs


def ultilty_df( user_history: pd.DataFrame) -> pd.DataFrame:
    # Ensure track_id and user_id are of type 'category'
    user_history["track_id"] = user_history["track_id"].astype("category")
    user_history["user_id"] = user_history["user_id"].astype("category")

    # Debug: Verify column types
    print(f"track_id dtype: {user_history['track_id'].dtype}")
    print(f"user_id dtype: {user_history['user_id'].dtype}")

    # Sort by user_id and track_id
    df = user_history.sort_values(["track_id", "user_id"]).copy()
    df.reset_index(drop=True, inplace=True)

    # Create a new column 'track_id_codes' to store the codes of the 'track_id' category

    df["track_id"] = df["track_id"].astype("category")
    df["user_id"] = df["user_id"].astype("category")

    df["track_id_codes"] = df["track_id"].cat.codes
    df["user_id_codes"] = df["user_id"].cat.codes


    return df

def save_item_user_matrix(cwd_path,utility_df: pd.DataFrame) -> None:

    interaction_matrix = utility_df.groupby(['track_id_codes', 'user_id_codes'],as_index=False)['playcount'].sum()

    row_indices = interaction_matrix['track_id_codes']
    col_indices = interaction_matrix['user_id_codes']
    values = interaction_matrix['playcount']
    n_tracks = interaction_matrix['track_id_codes'].nunique()
    n_users = interaction_matrix['user_id_codes'].nunique()

    sparse_matrix = csr_matrix((values, (row_indices, col_indices)), shape=(n_tracks, n_users))


    path_loc = cwd_path / "data" / "item_user_matrix.npz"
    # Save the sparse matrix to a file
    save_npz(path_loc, sparse_matrix)
    print("Sparse matrix saved to item_user_matrix.npz")

def collaborative_recommendation(song_name,artist_name,songs_data,interaction_matrix,k):
    song_row = songs_data[(songs_data["name"].str.lower()== song_name.lower()) & (songs_data["artist"].str.lower() == artist_name.lower())]
    if song_row.empty:
        print(f"Song '{song_name}' by '{artist_name}' not found.")
        return None
    
    songs_data['track_id'] = songs_data['track_id'].astype('category')
    # track_id of input song
    input_track_id = song_row['track_id'].values.item()
    print(input_track_id)
    # index value of track_id
    input_track_index = np.where(songs_data['track_id'].cat.categories == input_track_id)[0].item()
    print(input_track_index)


    input_track_vector = interaction_matrix[input_track_index]
    similarity_scores = cosine_similarity(input_track_vector, interaction_matrix)
    print("similarity score",similarity_scores)
    print("arg sort ",)
    # top scores
    recommendation_track_ids = np.argsort(-similarity_scores).ravel()[:k+1]
    print(recommendation_track_ids)
    print(songs_data.iloc[recommendation_track_ids])

    return (
      songs_data
      .iloc[recommendation_track_ids]
      .reset_index(drop=True)
  )




def main():
    cwd_path = Path.cwd()
    cleaned_songs_path = cwd_path / "data" / "cleaned_data.csv"
    user_history_path = cwd_path / "data" / "User Listening History.csv"

    


    # Load the cleaned songs data
    cleaned_songs = pd.read_csv(cleaned_songs_path)

    # Load the user listening history data
    user_history = pd.read_csv(user_history_path)


    # save the filtered songs data
    save_user_listening_data(cwd_path,cleaned_songs, user_history)

    # utility_df
    utility_df = ultilty_df(user_history)

    # item user matrix 
    save_item_user_matrix(cwd_path,utility_df)



    # too check the collaborative recommendation works
    

    # song_name = 'Love Story'
    # artist_name = 'Taylor Swift'
    # path_loc_item_user_mat = cwd_path / "data" / "item_user_matrix.npz"
    # path_loc_filtered_songs = cwd_path / "data" / "filtered_songs.csv"

    # #Load the sparse matrix
    # interaction_matrix = load_npz(path_loc_item_user_mat)
    # filtered_songs_df = pd.read_csv(path_loc_filtered_songs)

    # df_result = collaborative_recommendation(song_name,artist_name,
    #                          filtered_songs_df,
    #                          interaction_matrix,k=5)
    # print(df_result.head())



if __name__ == "__main__":
    main()