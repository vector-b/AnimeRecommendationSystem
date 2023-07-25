import streamlit as st
def main():
    st.set_page_config(
    page_title="How it Works?",
    page_icon="🤔",
    )
    with open("markdown/howitworks.md", "r", encoding="utf-8") as file:
        about_text = file.read()

    intro_text = '''
    ## BUT HOW IT WORKS?????

In this page, we will explain how our Anime Recommendation System works based on data from MyAnimeList (MAL). The goal of our project is to help users discover new anime based on their preferences and ratings from other users. The system uses collaborative filtering and text processing techniques to provide personalized and relevant recommendations.

    '''
    st.markdown(intro_text)
    thinking_img = "imgs\luffy-thinking.gif"
    st.image(thinking_img, caption='HM... YEAH THAT MAKES SENSE', use_column_width=True)

    # Exibindo o conteúdo do arquivo no Streamlit
    st.markdown(about_text)

    filtering_img = "imgs/filtering.png"
    st.image(filtering_img, caption='Types of Filtering', use_column_width=True)

if __name__ == "__main__":
    main()