import streamlit as st
import pandas as pd 
import numpy as np
import os
import difflib
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from models.data_manager import AnimeRecommendation, UserRecommendation

st.set_page_config(page_title="App",
                   page_icon="✨",
                   layout="centered",
                   initial_sidebar_state="expanded")

@st.cache_resource
def load_data():
    AniRec = AnimeRecommendation()
    UsrRec = UserRecommendation()
    AniRec.preprocess_data()
    UsrRec.preprocess_user_data()
    return AniRec, UsrRec

AniRec, UsrRec = load_data()



def app():


    st.title("Anime Recommendation System!")

    opt = st.radio(
        "Select mode:",
        ('Recommendation by Anime', 'User-Anime Relationship', 'User-Based Anime Recommendation'))

    if opt == 'Recommendation by Anime':
        AniRec.recommendation_by_anime()
    elif opt == 'User-Anime Relationship':
        UsrRec.get_user_taste_relation()
    elif opt == 'User-Based Anime Recommendation':
        UsrRec.get_user_list_recommendation()



    

    

if __name__ == "__main__":
    app()


