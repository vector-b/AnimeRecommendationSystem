import pandas as pd
import numpy as np 
import difflib
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import streamlit as st
import gc

# Define a class AnimeRecommendation
class AnimeRecommendation():
    
    # Initialize the class instance
    def __init__(self):

        # Read anime data from CSV files
        anime_list = pd.read_csv("data/anime.csv")
        anime_description = pd.read_csv("data/anime_with_synopsis.csv")

        # Merge the anime data based on the "MAL_ID" column
        self.anime_complete = pd.merge(anime_list, anime_description[["MAL_ID","sypnopsis"]], how='inner', on="MAL_ID")
        
        del anime_list
        del anime_description

        gc.collect()

        # Define the features to be used for recommendation
        self.features = ["Name", "Genres", "sypnopsis"]
        self.vectorizer = TfidfVectorizer()

    # Compute the general similarity matrix based on the feature vectors
    def get_general_similarity(self, data):
        return cosine_similarity(data)

    # Preprocess the anime data, handling missing values and creating feature vectors
    def preprocess_data(self):

        self.anime_complete.sypnopsis.fillna(' ', inplace=True)
        
        featured = self.anime_complete[self.features].apply(lambda row: '_'.join(row.values.astype(str)), axis=1)
        
        self.feature_vectors = self.vectorizer.fit_transform(featured)

        self.similarity = self.get_general_similarity(self.feature_vectors)

    # Compute the cosine similarity between two anime based on their IDs
    def get_cosine_similarity(self, anime_id_1, anime_id_2):
        anime_features = self.similarity

        idx_anime_1 = self.get_anime_index(anime_id_1)
        idx_anime_2 = self.get_anime_index(anime_id_2)

        if idx_anime_1 is not None and idx_anime_2 is not None:
            anime_vector_1 = self.feature_vectors[idx_anime_1]
            anime_vector_2 = self.feature_vectors[idx_anime_2]

            similarity = cosine_similarity([anime_vector_1], [anime_vector_2])[0][0]
            return similarity
        else:
            return None
    
    # Get the index of an anime based on its name, using fuzzy matching
    def get_anime_index_by_name(self, name):
        title_list = self.anime_complete.Name.to_list()
        match_list = difflib.get_close_matches(name, title_list)
        if not match_list:
            return None
        else:
            anime_match = match_list[0]
            idx_anime = self.anime_complete[self.anime_complete.Name == anime_match].index.values[0]
            return idx_anime, anime_match
        
    # Get the index of an anime based on its ID
    def get_anime_index_by_id(self, id):
        id_list = self.anime_complete.MAL_ID.to_list()
        match_filter = self.anime_complete[self.anime_complete.MAL_ID == id]
        if match_filter.empty:
            return None
        else:
            idx_anime = match_filter.index.values[0]
            return idx_anime
        
    # Get the description of an anime based on its name
    def get_anime_description_by_name(self, name):
        return self.anime_complete[self.anime_complete['Name'] == name]['sypnopsis'].iloc[0]
        
    # Get a list of similar animes based on the provided anime name and a specified number of results
    def similar_animes_by_name(self, name, number):
        idx_anime, anime_name = self.get_anime_index_by_name(name)
        if idx_anime is None:
            return None
        else:
            similarity_score = list(enumerate(self.similarity[idx_anime]))
            most_similar_animes = sorted(similarity_score, key=lambda x:x[1], reverse=True)
            anime_names = [self.anime_complete.loc[anime[0], 'Name'] for anime in most_similar_animes[:number]]
            anime_list = [{"nome": name, "descricao": self.get_anime_description_by_name(name)} for name in anime_names]
            
            return anime_list, anime_name
    
    # Show a list of anime recommendations in an expander widget in the Streamlit app
    def show_anime_list_expander(self, anime_list):
        if anime_list:
            st.write('Here are some anime recommendations based on the searched anime: ')
            for item in anime_list:
                with st.expander(item["nome"]):
                    st.write(item["descricao"])
        else:
            st.write('Anime not found, try another query...')

    # Provide anime recommendations based on user input in the Streamlit app
    def recommendation_by_anime(self):
        title = st.text_input('Please enter the name of the anime here: ')
        number = st.slider('Number of entries: ', 1, 30, 10)
        try:
            if title != "":
                anime_list, anime_name = self.similar_animes_by_name(title, number)
                st.info(f"Anime found: {anime_name}")
                self.show_anime_list_expander(anime_list)
        except Exception as e:
            st.write('Anime not found, try another query...')

# Define a subclass UserRecommendation that inherits from AnimeRecommendation
class UserRecommendation(AnimeRecommendation):
    def __init__(self):
        # Call the constructor of the parent class to initialize common attributes
        AnimeRecommendation.__init__(self)
        
        # Define the maximum number of users to consider
        MAX_USERS = 1000
        # Read user rating data from CSV file
        user_rating = pd.read_csv("data/animelist.csv").head(200000).sort_values(by=["user_id","anime_id"])

        # Select a subset of unique users randomly
        unique_users = user_rating['user_id'].unique()
        unique_users = np.random.choice(unique_users, size=min(MAX_USERS, len(unique_users)), replace=False)
        unique_users = np.sort(unique_users)

        self.unique_users = unique_users
        self.unique_anime = user_rating.anime_id.unique()

        self.user_rating = user_rating[user_rating['user_id'].isin(unique_users)]

        del user_rating

        gc.collect()
    
    # Compute the mean user rating for each user
    def get_mean_user_rating(self, df):
        mean = df.groupby(by="user_id",as_index=False)['rating'].mean()
        return mean
    
    # Preprocess the user rating data and create a user-item rating matrix
    def preprocess_user_data(self):

        Mean = self.get_mean_user_rating(self.user_rating)
        
        Rating_avg = pd.merge(self.user_rating, Mean, on='user_id')
        Rating_avg['avg_rating']=Rating_avg['rating_x']-Rating_avg['rating_y']

        self.mean = Mean

        self.user_pivot_table = pd.pivot_table(Rating_avg, values='avg_rating',index='user_id',columns='anime_id')
        self.user_rating = self.user_pivot_table.fillna(self.user_pivot_table.mean(axis=0))

        Rating_avg