import os
import logging
import streamlit as st
from streamlit_card import card
from demo_util import DemoFileIOHelper, DemoUIHelper, get_demo_dir, display_article_page

# Configureer logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Voeg een handler toe om logs naar een bestand te schrijven
file_handler = logging.FileHandler('my_articles_page.log')
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
logger.addHandler(file_handler)

def my_articles_page():
    logger.info("Starting my_articles_page function")
    image_path = os.path.join(get_demo_dir(), "assets", "void.jpg")
    logger.debug(f"Image path set to: {image_path}")

    with st.sidebar:
        _, return_button_col = st.columns([2, 5])
        with return_button_col:
            if st.button("Select another article", disabled="page2_selected_my_article" not in st.session_state):
                logger.info("User clicked 'Select another article' button")
                if "page2_selected_my_article" in st.session_state:
                    del st.session_state["page2_selected_my_article"]
                    logger.debug("Removed page2_selected_my_article from session state")
                st.rerun()

    if "page2_user_articles_file_path_dict" not in st.session_state:
        logger.info("Initializing page2_user_articles_file_path_dict")
        local_dir = os.path.join(get_demo_dir(), "DEMO_WORKING_DIR")
        os.makedirs(local_dir, exist_ok=True)
        logger.debug(f"Created local directory: {local_dir}")
        st.session_state["page2_user_articles_file_path_dict"] = DemoFileIOHelper.read_structure_to_dict(local_dir)
        logger.debug("Read directory structure to dictionary")

    def article_card_setup(column_to_add, card_title, article_name):
        logger.debug(f"Setting up article card for: {article_name}")
        with column_to_add:
            cleaned_article_title = article_name.replace("_", " ")
            image_path = os.path.join(get_demo_dir(), "assets", "void.jpg")
            try:
                image_data = DemoFileIOHelper.read_image_as_base64(image_path)
                logger.debug("Successfully read image data")
            except Exception as e:
                logger.error(f"Failed to read image data: {str(e)}")
                image_data = None

            hasClicked = card(title=" / ".join(card_title),
                              text=cleaned_article_title,
                              image=image_data,
                              styles=DemoUIHelper.get_article_card_UI_style(boarder_color="#9AD8E1"))
            if hasClicked:
                logger.info(f"User clicked on article card: {article_name}")
                st.session_state["page2_selected_my_article"] = article_name
                st.rerun()

    if "page2_selected_my_article" not in st.session_state:
        logger.info("No article selected, displaying article list")
        my_article_columns = st.columns(3)
        if len(st.session_state["page2_user_articles_file_path_dict"]) > 0:
            article_names = sorted(list(st.session_state["page2_user_articles_file_path_dict"].keys()))
            logger.debug(f"Found {len(article_names)} articles")
            pagination = st.container()
            bottom_menu = st.columns((1, 4, 1, 1, 1))[1:-1]
            with bottom_menu[2]:
                batch_size = st.selectbox("Page Size", options=[24, 48, 72])
                logger.debug(f"User selected batch size: {batch_size}")
            with bottom_menu[1]:
                total_pages = max(1, int(len(article_names) / batch_size))
                current_page = st.number_input("Page", min_value=1, max_value=total_pages, step=1)
                logger.debug(f"Current page: {current_page} of {total_pages}")
            with bottom_menu[0]:
                st.markdown(f"Page **{current_page}** of **{total_pages}** ")
            with pagination:
                start_index = (current_page - 1) * batch_size
                end_index = min(current_page * batch_size, len(article_names))
                logger.debug(f"Displaying articles from index {start_index} to {end_index}")
                for idx, article_name in enumerate(article_names[start_index:end_index]):
                    column_to_add = my_article_columns[idx % 3]
                    article_card_setup(column_to_add=column_to_add,
                                       card_title=["My Article"],
                                       article_name=article_name)
        else:
            logger.info("No articles found, displaying 'Get started' card")
            with my_article_columns[0]:
                image_path = os.path.join(get_demo_dir(), "assets", "void.jpg")
                try:
                    image_data = DemoFileIOHelper.read_image_as_base64(image_path)
                    logger.debug("Successfully read image data for 'Get started' card")
                except Exception as e:
                    logger.error(f"Failed to read image data for 'Get started' card: {str(e)}")
                    image_data = None

                hasClicked = card(title="Get started",
                                  text="Start your first research!",
                                  image=image_data,
                                  styles=DemoUIHelper.get_article_card_UI_style())
                if hasClicked:
                    logger.info("User clicked 'Get started' card")
                    st.session_state.selected_page = 1
                    st.session_state["manual_selection_override"] = True
                    st.session_state["rerun_requested"] = True
                    st.rerun()
    else:
        logger.info("Article selected, displaying article page")
        selected_article_name = st.session_state["page2_selected_my_article"]
        selected_article_file_path_dict = st.session_state["page2_user_articles_file_path_dict"][selected_article_name]
        logger.debug(f"Displaying article: {selected_article_name}")

        display_article_page(selected_article_name=selected_article_name,
                             selected_article_file_path_dict=selected_article_file_path_dict,
                             show_title=True, show_main_article=True)

logger.info("my_articles_page module initialized and logging configured")