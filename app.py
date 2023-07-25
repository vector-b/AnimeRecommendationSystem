import streamlit as st
import pandas as pd 
import numpy as np
import os
import difflib
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from models.data_manager import AnimeRecommendation, UserRecommendation

@st.cache_resource
def load_data():
    AniRec = AnimeRecommendation()
    UsrRec = UserRecommendation()
    AniRec.preprocess_data()
    UsrRec.preprocess_user_data()
    return AniRec, UsrRec

AniRec, UsrRec = load_data()



def app():
    
    st.title("Sistema de Recomendação de Animes!")

    opt = st.radio(
    "Selecione o modo: ",
    ('Anime', 'User'))

    if opt == 'Anime':
        AniRec.recommendation_by_anime()
    elif opt == 'User':
        UsrRec.get_user_taste_relation()

    

    

if __name__ == "__main__":
    app()


