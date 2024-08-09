import os
import time
import logging

import demo_util
import streamlit as st
from demo_util import DemoFileIOHelper, DemoTextProcessingHelper, DemoUIHelper

# Configureer logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Voeg een handler toe om logs naar een bestand te schrijven
file_handler = logging.FileHandler('create_new_article.log')
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
logger.addHandler(file_handler)

def create_new_article_page():
    logger.info("Starting create_new_article_page function")
    demo_util.clear_other_page_session_state(page_index=3)
    logger.debug("Cleared other page session states")

    if "page3_write_article_state" not in st.session_state:
        st.session_state["page3_write_article_state"] = "not started"
        logger.debug("Initialized page3_write_article_state")

    if st.session_state["page3_write_article_state"] == "not started":
        logger.info("Article writing process not started yet")
        _, search_form_column, _ = st.columns([2, 5, 2])
        with search_form_column:
            with st.form(key='search_form'):
                DemoUIHelper.st_markdown_adjust_size(content="Enter the topic you want to learn in depth:",
                                                     font_size=18)
                st.session_state["page3_topic"] = st.text_input(label='page3_topic', label_visibility="collapsed")
                pass_appropriateness_check = True

                submit_button = st.form_submit_button(label='Research')
                if submit_button and st.session_state["page3_write_article_state"] in ["not started", "show results"]:
                    logger.info(f"Research button clicked for topic: {st.session_state['page3_topic']}")
                    if not st.session_state["page3_topic"].strip():
                        pass_appropriateness_check = False
                        st.session_state["page3_warning_message"] = "topic could not be empty"
                        logger.warning("Empty topic submitted")

                    st.session_state["page3_topic_name_cleaned"] = st.session_state["page3_topic"].replace(
                        ' ', '_').replace('/', '_')
                    if not pass_appropriateness_check:
                        st.session_state["page3_write_article_state"] = "not started"
                        alert = st.warning(st.session_state["page3_warning_message"], icon="⚠️")
                        logger.warning(f"Appropriateness check failed: {st.session_state['page3_warning_message']}")
                        time.sleep(5)
                        alert.empty()
                    else:
                        st.session_state["page3_write_article_state"] = "initiated"
                        logger.info("Article writing process initiated")

    if st.session_state["page3_write_article_state"] == "initiated":
        logger.info("Setting up working directory")
        current_working_dir = os.path.join(demo_util.get_demo_dir(), "DEMO_WORKING_DIR")
        if not os.path.exists(current_working_dir):
            os.makedirs(current_working_dir)
            logger.debug(f"Created working directory: {current_working_dir}")

        if "runner" not in st.session_state:
            demo_util.set_storm_runner()
            logger.debug("Set STORM runner")
        st.session_state["page3_current_working_dir"] = current_working_dir
        st.session_state["page3_write_article_state"] = "pre_writing"
        logger.info("Moved to pre-writing state")

    if st.session_state["page3_write_article_state"] == "pre_writing":
        logger.info("Starting pre-writing phase")
        status = st.status("I am brain**STORM**ing now to research the topic. (This may take 2-3 minutes.)")
        st_callback_handler = demo_util.StreamlitCallbackHandler(status)
        with status:
            logger.debug("Running STORM for research and outline generation")
            # Eerst You.com gebruiken
            st.session_state["runner"].retriever.set_active_retrievers(['you'])
            st.session_state["runner"].run(
                topic=st.session_state["page3_topic"],
                do_research=True,
                do_generate_outline=True,
                do_generate_article=False,
                do_polish_article=False,
                callback_handler=st_callback_handler
            )
            # Daarna vector retriever gebruiken
            st.session_state["runner"].retriever.set_active_retrievers(['vector'])
            st.session_state["runner"].run(
                topic=st.session_state["page3_topic"],
                do_research=True,
                do_generate_outline=False,
                do_generate_article=False,
                do_polish_article=False,
                callback_handler=st_callback_handler
            )
            conversation_log_path = os.path.join(st.session_state["page3_current_working_dir"],
                                                 st.session_state["page3_topic_name_cleaned"], "conversation_log.json")
            demo_util._display_persona_conversations(DemoFileIOHelper.read_json_file(conversation_log_path))
            logger.debug("Displayed persona conversations")
            st.session_state["page3_write_article_state"] = "final_writing"
            status.update(label="brain**STORM**ing complete!", state="complete")
            logger.info("Pre-writing phase completed")

    if st.session_state["page3_write_article_state"] == "final_writing":
        logger.info("Starting final writing phase")
        with st.status(
                "Now I will connect the information I found for your reference. (This may take 4-5 minutes.)") as status:
            st.info('Now I will connect the information I found for your reference. (This may take 4-5 minutes.)')
            logger.debug("Running STORM for article generation and polishing")
            st.session_state["runner"].retriever.set_active_retrievers(['you', 'vector'])
            st.session_state["runner"].run(topic=st.session_state["page3_topic"], do_research=False,
                                           do_generate_outline=False,
                                           do_generate_article=True, do_polish_article=True, remove_duplicate=False)
            st.session_state["runner"].post_run()
            logger.debug("STORM run completed")

            st.session_state["page3_write_article_state"] = "prepare_to_show_result"
            status.update(label="information synthesis complete!", state="complete")
            logger.info("Final writing phase completed")

    if st.session_state["page3_write_article_state"] == "prepare_to_show_result":
        logger.info("Preparing to show results")
        _, show_result_col, _ = st.columns([4, 3, 4])
        with show_result_col:
            if st.button("show final article"):
                st.session_state["page3_write_article_state"] = "completed"
                logger.debug("User clicked to show final article")
                st.rerun()

    if st.session_state["page3_write_article_state"] == "completed":
        logger.info("Displaying final article")
        current_working_dir_paths = DemoFileIOHelper.read_structure_to_dict(
            st.session_state["page3_current_working_dir"])
        current_article_file_path_dict = current_working_dir_paths[st.session_state["page3_topic_name_cleaned"]]
        demo_util.display_article_page(selected_article_name=st.session_state["page3_topic_name_cleaned"],
                                       selected_article_file_path_dict=current_article_file_path_dict,
                                       show_title=True, show_main_article=True)
        logger.debug("Article displayed")

logger.info("create_new_article_page module initialized and logging configured")