import streamlit as st
from content_based_filtering import recommend
from scipy.sparse import load_npz
import pandas as pd
from collaberative_filtering import collaborative_recommendation
from hybrid_recomender import hybrid_rec
from pathlib import Path


# path
transformed_data_path = "data/transformed_data.npz"
cleaned_data_path = "data/cleaned_data.csv"
path_item_user_mat_collab = "data/item_user_matrix.npz"
path_filtered_songs = "data/filtered_songs.csv"
path_mat_content = "data/content_based.npz"

# load the data

st.session_state.songs_data = pd.read_csv(cleaned_data_path)  
st.session_state.filtered_songs_df = pd.read_csv(path_filtered_songs)
st.session_state.transformed_data = load_npz(transformed_data_path)
st.session_state.collab_interaction_matrix = load_npz(path_item_user_mat_collab)
st.session_state.content_interaction_matrix = load_npz(path_mat_content)

# Title
st.title('Welcome to the Spotify Song Recommender!')

# Subheader
st.write('### Enter the name of a song and the recommender will suggest similar songs 🎵🎧')

# Text Input
song_name = st.text_input('Enter a song name:')
st.write('You entered:', song_name)
# artist name
artist_name = st.text_input('Enter the artist name:')
st.write('You entered:', artist_name)
# lowercase the input
song_name = song_name.lower()
artist_name = artist_name.lower()

# k recommendations
k = st.selectbox('How many recommendations do you want?', [5,10,15,20], index=1)

# Filtering type selection
if ((st.session_state.filtered_songs_df["name"].str.lower() == song_name.lower()) & (st.session_state.filtered_songs_df["artist"] == artist_name.lower())).any():   
    # type of filtering
    filtering_type = st.selectbox(label= 'Select the type of filtering:', 
                                options= ['Content-Based Filtering', 
                                          'Collaborative Filtering',
                                          "Hybrid Recommender System"],
                                index= 2)
    
    # Diversity slider only appears for Hybrid Recommender System
    if filtering_type == "Hybrid Recommender System":
        diversity = st.slider(label="Diversity in Recommendations",
                            min_value=1,
                            max_value=10,
                            value=5,
                            step=1)
        content_based_weight = 1 - (diversity / 10)
        collaborative_weight = diversity/10
    else:
        # Default weights for other filtering types
        content_based_weight = 1
        collaborative_weight = 0

else:
    # type of filtering
    filtering_type = st.selectbox(label= 'Select the type of filtering:', 
                                options= ['Content-Based Filtering'])
    # Default weights
    content_based_weight = 1
    collaborative_weight = 0

# Button
if filtering_type == 'Content-Based Filtering':
    if st.button('Get Recommendations'):
        if ((st.session_state.songs_data["name"].str.lower() == song_name.lower()) & (st.session_state.songs_data["artist"].str.lower() == artist_name.lower())).any():
            st.write('Recommendations for', f"**{song_name}** by **{artist_name}**")
            recommendations = recommend(song_name, artist_name,
                                         st.session_state.songs_data, 
                                         st.session_state.transformed_data, k)
            
            # Display Recommendations
            for ind , recommendation in recommendations.iterrows():
                song_name = recommendation['name'].title()
                artist_name = recommendation['artist'].title()
                
                if ind == 0:
                    st.markdown("## Currently Playing")
                    st.markdown(f"#### **{song_name}** by **{artist_name}**")
                    st.audio(recommendation['spotify_preview_url'])
                    st.write('---')
                elif ind == 1:   
                    st.markdown("### Next Up 🎵")
                    st.markdown(f"#### {ind}. **{song_name}** by **{artist_name}**")
                    st.audio(recommendation['spotify_preview_url'])
                    st.write('---')
                else:
                    st.markdown(f"#### {ind}. **{song_name}** by **{artist_name}**")
                    st.audio(recommendation['spotify_preview_url'])
                    st.write('---')
        else:
            st.write(f"Sorry, we couldn't find {song_name} in our database. Please try another song.")
            
elif filtering_type == 'Collaborative Filtering':
    if st.button('Get Recommendations'):
        if ((st.session_state.filtered_songs_df["name"].str.lower() == song_name.lower()) & 
    (st.session_state.filtered_songs_df["artist"].str.lower() == artist_name.lower())).any():
            st.write('Recommendations for', f"**{song_name}** by **{artist_name}**")
            recommendations = collaborative_recommendation(song_name,
                                            artist_name,
                                          st.session_state.filtered_songs_df,
                                          st.session_state.collab_interaction_matrix,
                                          k=k)
            
            # Display Recommendations
            for ind , recommendation in recommendations.iterrows():
                song_name = recommendation['name'].title()
                artist_name = recommendation['artist'].title()
                
                if ind == 0:
                    st.markdown("## Currently Playing")
                    st.markdown(f"#### **{song_name}** by **{artist_name}**")
                    st.audio(recommendation['spotify_preview_url'])
                    st.write('---')
                elif ind == 1:   
                    st.markdown("### Next Up 🎵")
                    st.markdown(f"#### {ind}. **{song_name}** by **{artist_name}**")
                    st.audio(recommendation['spotify_preview_url'])
                    st.write('---')
                else:
                    st.markdown(f"#### {ind}. **{song_name}** by **{artist_name}**")
                    st.audio(recommendation['spotify_preview_url'])
                    st.write('---')
        else:
            st.write(f"Sorry, we couldn't find {song_name} in our database. Please try another song.")

elif filtering_type == "Hybrid Recommender System":
    if st.button('Get Recommendations'):
        if ((st.session_state.filtered_songs_df["name"].str.lower() == song_name.lower()) & 
    (st.session_state.filtered_songs_df["artist"].str.lower() == artist_name.lower())).any():

            st.write('Recommendations for', f"**{song_name}** by **{artist_name}**")
                                    
            # get the recommendations
            recommendations = hybrid_rec(song_name, artist_name, st.session_state.filtered_songs_df, 
                                             st.session_state.collab_interaction_matrix, 
                                             st.session_state.content_interaction_matrix,
                                             k, content_based_weight, collaborative_weight)
            # Display Recommendations
            for ind , recommendation in recommendations.iterrows():
                song_name = recommendation['name'].title()
                artist_name = recommendation['artist'].title()
                
                if ind == 0:
                    st.markdown("## Currently Playing")
                    st.markdown(f"#### **{song_name}** by **{artist_name}**")
                    st.audio(recommendation['spotify_preview_url'])
                    st.write('---')
                elif ind == 1:   
                    st.markdown("### Next Up 🎵")
                    st.markdown(f"#### {ind}. **{song_name}** by **{artist_name}**")
                    st.audio(recommendation['spotify_preview_url'])
                    st.write('---')
                else:
                    st.markdown(f"#### {ind}. **{song_name}** by **{artist_name}**")
                    st.audio(recommendation['spotify_preview_url'])
                    st.write('---')
        else:
            st.write(f"Sorry, we couldn't find {song_name} in our database. Please try another song.")