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
    return AniRec

AniRec = load_data()

def app():
    AniRec.preprocess_data()
    
    st.title("Sistema de Recomendação de Animes!")

    title = st.text_input('Digite o nome do anime: ')
    number = st.slider('Número de Animes: ', 1, 30, 10)

    if title != "":
        anime_list = AniRec.similar_animes_by_name(title, number)
        st.write(anime_list)

    

if __name__ == "__main__":
    app()


