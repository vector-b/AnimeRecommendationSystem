import pandas as pd
import numpy as np 
import difflib
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import streamlit as st
import gc

class AnimeRecommendation():
    
    def __init__(self):

        
        anime_list = pd.read_csv("data/anime.csv")
        anime_description = pd.read_csv("data/anime_with_synopsis.csv")

        self.anime_complete = pd.merge(anime_list, anime_description[["MAL_ID","sypnopsis"]], how='inner', on="MAL_ID")
        
        del anime_list
        del anime_description

        gc.collect()


        #self.features =  ["Name", "Genres", "Episodes","Studios","Rating","Score","Aired","sypnopsis"]
        self.features = ["Name", "Genres", "sypnopsis"]
        self.vectorizer = TfidfVectorizer()

    def get_general_similarity(self, data):
        return cosine_similarity(data)

    def preprocess_data(self):

        self.anime_complete.sypnopsis.fillna(' ', inplace=True)
        
        featured = self.anime_complete[self.features].apply(lambda row: '_'.join(row.values.astype(str)), axis=1)
        
        self.feature_vectors = self.vectorizer.fit_transform(featured)

        self.similarity = self.get_general_similarity(self.feature_vectors)

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

class UserRecommendation(AnimeRecommendation):
    def __init__(self):

        MAX_USERS = 100
        user_rating = pd.read_csv("data/animelist.csv").head(100000).sort_values(by=["user_id","anime_id"])
        unique_users = user_rating['user_id'].unique()[:MAX_USERS]
        
        self.unique_users = unique_users
        self.unique_anime = user_rating.anime_id.unique()

        self.user_rating = user_rating[user_rating['user_id'].isin(unique_users)]

        del user_rating

        gc.collect()
    
    def get_mean_user_rating(self, df):
        mean = df.groupby(by="user_id",as_index=False)['rating'].mean()
        return mean
    
    def preprocess_user_data(self):

        Mean = self.get_mean_user_rating(self.user_rating)
        self.mean = Mean
        Rating_avg = pd.merge(self.user_rating, Mean, on='user_id')
        Rating_avg['avg_rating']=Rating_avg['rating_x']-Rating_avg['rating_y']


        self.user_rating = pd.pivot_table(Rating_avg, values='avg_rating',index='user_id',columns='anime_id')
        self.user_rating = self.user_rating.fillna(self.user_rating.mean(axis=0))

        self.cosine = self.get_general_similarity(self.user_rating)
        np.fill_diagonal(self.cosine,0)

        self.square_rating = pd.DataFrame(self.cosine,index=self.user_rating.index)

    def get_most_similar_users(self, number):
        #order = np.argsort(self.square_rating, axis=1[:,:number])
        df = self.square_rating.apply(lambda x: pd.Series(x.sort_values(ascending=False)
           .iloc[:number].index, 
          index=['{}'.format(i) for i in range(1, number+1)]), axis=1)
        return df 

    def user_item_score(self, user, item):

        next_30_users = self.get_most_similar_users(30)
        a = next_30_users[next_30_users.index==user].values

        b = a.squeeze().tolist()
        c = self.user_rating.loc[:,item]
        d = c[c.index.isin(b)]

        #Only the users with the same anime 
        matched_users = d[d.notnull()]

        index = matched_users.index.values.squeeze().tolist()

        corr = self.square_rating.iloc[user, index]

        # Cria um DataFrame com as classificações e as similaridades
        user_similarity_df = pd.concat([matched_users, corr], axis=1)
        user_similarity_df.columns = ['avg_score', 'correlation']

        # Calcula o escore final do usuário para o item usando a fórmula de filtragem colaborativa
        numerator = (user_similarity_df['avg_score'] * user_similarity_df['correlation']).sum()
        denominator = user_similarity_df['correlation'].sum()


        mean = self.mean
    
        avg_user = mean.loc[mean['user_id'] == user,'rating'].values[0]

        final_score = avg_user + (numerator / denominator)
        return final_score

    def get_user_taste_relation(self):
        unique_users = self.unique_users
        unique_anime = self.unique_anime
        input_value_1 = st.select_slider(
            'Selecione a  1ª entrada: ',
        options=unique_users)

        input_value_2 = st.select_slider(
            'Selecione a  2ª entrada: ',
        options=unique_anime)
        
        st.write(self.user_rating)
        st.write(f"Relação de gosto de usuários {self.user_item_score(input_value_1, input_value_2)}")

