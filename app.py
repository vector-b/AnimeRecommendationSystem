import streamlit as st
import pandas as pd 
import numpy as np
import os
import difflib
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from models.data_manager import AnimeRecommendation

@st.cache_resource
def load_data():
    AniRec = AnimeRecommendation()
    AniRec.preprocess_data()
    return AniRec

AniRec = load_data()



def app():
    
    st.title("Sistema de Recomendação de Animes!")

    opt = st.radio(
    "Selecione o modo: ",
    ('Anime', 'User'))

    if opt == 'Anime':
        AniRec.recommendation_by_anime()
    elif opt == 'User':
        st.write("In progress...")
    else:
        st.write("Selecione um modo...")

    

    

if __name__ == "__main__":
    app()


