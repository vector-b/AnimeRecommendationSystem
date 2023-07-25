import streamlit as st
def main():
    st.set_page_config(
    page_title="About",
    page_icon="👋",
    )

    st.title('Project Info')
    st.write('In this project, the goal is to create a user-based recommendation system using reviews from MyAnimeList.')
    st.markdown("The dataset is available here: [Anime Recommendation Database](https://www.kaggle.com/datasets/hernan4444/anime-recommendation-database-2020)")
    st.markdown("Developed by: [vector-b](https://github.com/vector-b)")

if __name__ == "__main__":
    main()