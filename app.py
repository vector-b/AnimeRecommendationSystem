import streamlit as st
from models.anime_recommender_utils import AnimeRecommendation, UserRecommendation

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


