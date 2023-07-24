import pandas as pd
import numpy as np 
import difflib
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import streamlit as st
import gc

class AnimeRecommendation():
    
    def __init__(self):

        #anime_data = pd.read_csv("data/animelist.csv")
        anime_list = pd.read_csv("data/anime.csv")
        anime_description = pd.read_csv("data/anime_with_synopsis.csv")

        self.anime_complete = pd.merge(anime_list, anime_description[["MAL_ID","sypnopsis"]], how='inner', on="MAL_ID")
        
        #del anime_dat
        del anime_list
        del anime_description
  
        gc.collect()
        #self.features =  ["Name", "Genres", "Episodes","Studios","Rating","Score","Aired","sypnopsis"]
        self.features = ["Name", "Genres", "sypnopsis"]
        self.vectorizer = TfidfVectorizer()

    def preprocess_data(self):

        self.anime_complete.sypnopsis.fillna(' ', inplace=True)
        
        featured = self.anime_complete[self.features].apply(lambda row: '_'.join(row.values.astype(str)), axis=1)
        
        self.feature_vectors = self.vectorizer.fit_transform(featured)

        self.similarity = cosine_similarity(self.feature_vectors)

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
    
    def get_anime_index_by_name(self, name):
        title_list = self.anime_complete.Name.to_list()
        match_list = difflib.get_close_matches(name, title_list)
        if not match_list:
            return None
        else:
            anime_match = match_list[0]
            idx_anime = self.anime_complete[self.anime_complete.Name == anime_match].index.values[0]
            return idx_anime
        
    def get_anime_index_by_id(self, id):
        id_list = self.anime_complete.MAL_ID.to_list()
        match_filter = self.anime_complete[self.anime_complete.MAL_ID == id]
        if match_filter.empty:
            return None
        else:
            idx_anime = match_filter.index.values[0]
            return idx_anime
        
    def get_anime_description_by_name(self, name):
        return self.anime_complete[self.anime_complete['Name'] == name]['sypnopsis'].iloc[0]
        
    def similar_animes_by_name(self, name, number):
        idx_anime = self.get_anime_index_by_name(name)
        if idx_anime is None:
            return None
        else:
            similarity_score = list(enumerate(self.similarity[idx_anime]))
            most_similar_animes = sorted(similarity_score, key=lambda x:x[1], reverse=True)
            anime_names = [self.anime_complete.loc[anime[0], 'Name'] for anime in most_similar_animes[:number]]
            anime_list = [{"nome": name, "descricao": self.get_anime_description_by_name(name)} for name in anime_names]
            
            return anime_list
        
    def recommendation_by_anime(self):
        title = st.text_input('Digite o nome do anime: ')
        number = st.slider('Número de Animes: ', 1, 30, 10)

        if title != "":
            anime_list = self.similar_animes_by_name(title, number)
            
            if anime_list:
                st.write('Aqui estão algumas recomendações de anime com base no anime escolhido: ')
                for item in anime_list:
                    with st.expander(item["nome"]):
                        st.write(item["descricao"])
            else:
                st.write('O anime desejado não foi encontrado, tente novamente...')

