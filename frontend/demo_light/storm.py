import os

# Bepaal het pad naar de huidige map en de hoofdmap van de wiki
script_dir = os.path.dirname(os.path.abspath(__file__))
wiki_root_dir = os.path.dirname(os.path.dirname(script_dir))

import demo_util
from pages_util import MyArticles, CreateNewArticle
from streamlit_float import *
from streamlit_option_menu import option_menu


def main():
    global database
    st.set_page_config(layout='wide')

    # Initialiseer de sessie bij de eerste uitvoering
    if "first_run" not in st.session_state:
        st.session_state['first_run'] = True

    # Stel API-sleutels in vanuit geheimen bij de eerste uitvoering
    if st.session_state['first_run']:
        for key, value in st.secrets.items():
            if type(value) == str:
                os.environ[key] = value

    # Initialiseer sessievariabelen als ze nog niet bestaan
    if "selected_article_index" not in st.session_state:
        st.session_state["selected_article_index"] = 0
    if "selected_page" not in st.session_state:
        st.session_state["selected_page"] = 0
    
    # Voer de applicatie opnieuw uit als daarom wordt gevraagd
    if st.session_state.get("rerun_requested", False):
        st.session_state["rerun_requested"] = False
        st.rerun()

    # Pas de opmaak van de pagina aan
    st.write('<style>div.block-container{padding-top:2rem;}</style>', unsafe_allow_html=True)
    
    # Maak een container voor het menu
    menu_container = st.container()
    with menu_container:
        pages = ["My Articles", "Create New Article"]
        styles={
            "container": {"padding": "0.2rem 0", 
                          "background-color": "#22222200"},
        }
        
        # Maak een horizontaal menu met iconen
        menu_selection = option_menu(None, pages,
                                     icons=['house', 'search'],
                                     menu_icon="cast", default_index=0, orientation="horizontal",
                                     manual_select=st.session_state.selected_page,
                                     styles=styles,
                                     key='menu_selection')
        
        # Overschrijf de menuselectie indien nodig
        if st.session_state.get("manual_selection_override", False):
            menu_selection = pages[st.session_state["selected_page"]]
            st.session_state["manual_selection_override"] = False
            st.session_state["selected_page"] = None

        # Toon de geselecteerde pagina
        if menu_selection == "My Articles":
            demo_util.clear_other_page_session_state(page_index=2)
            MyArticles.my_articles_page()
        elif menu_selection == "Create New Article":
            demo_util.clear_other_page_session_state(page_index=3)
            CreateNewArticle.create_new_article_page()


if __name__ == "__main__":
    main()
