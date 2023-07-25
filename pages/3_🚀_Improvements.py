import streamlit as st

def main():
    st.set_page_config(
        page_title="Improvements",
        page_icon="🚀",
        )

    with open("markdown/howitworks.md", "r", encoding="utf-8") as file:
        improvements_text = file.read()

    st.markdown(improvements_text)

if __name__ == '__main__':
    main()


        