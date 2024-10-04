import os
import time

import demo_util
import streamlit as st
from demo_util import DemoFileIOHelper, DemoTextProcessingHelper, DemoUIHelper, truncate_filename

def handle_not_started():
    if st.session_state["page3_write_article_state"] == "not started":
        _, search_form_column, _ = st.columns([2, 5, 2])
        with search_form_column:
            with st.form(key='search_form'):
                DemoUIHelper.st_markdown_adjust_size(content="Enter the topic you want to learn in depth:", font_size=18)
                st.session_state["page3_topic"] = st.text_input(label='page3_topic', label_visibility="collapsed")
                st.session_state["use_costorm"] = st.checkbox("Use Co-STORM")
                pass_appropriateness_check = True

                submit_button = st.form_submit_button(label='Research')
                if submit_button and st.session_state["page3_write_article_state"] in ["not started", "show results"]:
                    if not st.session_state["page3_topic"].strip():
                        pass_appropriateness_check = False
                        st.session_state["page3_warning_message"] = "topic could not be empty"

                    st.session_state["page3_topic_name_cleaned"] = st.session_state["page3_topic"].replace(' ', '_').replace('/', '_')
                    st.session_state["page3_topic_name_truncated"] = truncate_filename(st.session_state["page3_topic_name_cleaned"])
                    if not pass_appropriateness_check:
                        st.session_state["page3_write_article_state"] = "not started"
                        alert = st.warning(st.session_state["page3_warning_message"], icon="⚠️")
                        time.sleep(5)
                        alert.empty()
                    else:
                        st.session_state["page3_write_article_state"] = "initiated"

def handle_initiated():
    if st.session_state["page3_write_article_state"] == "initiated":
        # Maak een werkmap aan als deze nog niet bestaat
        current_working_dir = os.path.join(demo_util.get_demo_dir(), "DEMO_WORKING_DIR")
        if not os.path.exists(current_working_dir):
            os.makedirs(current_working_dir)

        # Initialiseer de 'runner' als deze nog niet bestaat
        if "runner" not in st.session_state:
            demo_util.set_storm_runner()
        st.session_state["page3_current_working_dir"] = current_working_dir
        st.session_state["page3_write_article_state"] = "pre_writing"

def handle_costorm_interaction():
    if st.session_state["page3_write_article_state"] == "costorm_interaction":
        st.write("Co-STORM Interaction")
        
        # Toon de huidige mind map
        if "mind_map" in st.session_state:
            st.subheader("Current Mind Map")
            st.json(st.session_state["mind_map"])
        
        # Toon de conversatiegeschiedenis
        if "conversation_history" in st.session_state:
            st.subheader("Conversation History")
            for turn in st.session_state["conversation_history"]:
                st.text(f"{turn['speaker']}: {turn['utterance']}")
        
        # Gebruikersinvoer
        user_input = st.text_input("Enter your question or comment:")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("Send"):
                if user_input:
                    conv_turn = st.session_state["runner"].step(user_utterance=user_input)
                else:
                    conv_turn = st.session_state["runner"].step()
                
                # Update sessie-state met nieuwe informatie
                st.session_state["conversation_history"] = st.session_state["runner"].conversation_history
                st.session_state["mind_map"] = st.session_state["runner"].knowledge_base.to_dict()
                
                st.write("Co-STORM response:", conv_turn)
                st.rerun()
        
        with col2:
            if st.button("Show Mind Map"):
                st.json(st.session_state["runner"].knowledge_base.to_dict())
        
        with col3:
            if st.button("Generate report"):
                st.session_state["runner"].knowledge_base.reogranize()
                article = st.session_state["runner"].generate_report()
                st.session_state["costorm_article"] = article
                st.session_state["page3_write_article_state"] = "completed"
                st.rerun()

def handle_pre_writing():
    if st.session_state["page3_write_article_state"] == "pre_writing":
        status = st.status("I am brain**STORM**ing now to research the topic. (This may take 2-3 minutes.)")
        st_callback_handler = demo_util.StreamlitCallbackHandler(status)
        with status:
            if "runner" not in st.session_state:
                demo_util.set_storm_runner()
            
            st.session_state["runner"].run(
                topic=st.session_state["page3_topic"],
                do_research=True,
                do_generate_outline=True,
                do_generate_article=False,
                do_polish_article=False,
                callback_handler=st_callback_handler
            )
            
            conversation_log_path = os.path.join(st.session_state["page3_current_working_dir"],
                                                 st.session_state["page3_topic_name_truncated"], "conversation_log.json")
            
            if os.path.exists(conversation_log_path):
                demo_util._display_persona_conversations(DemoFileIOHelper.read_json_file(conversation_log_path))
            else:
                st.error(f"Conversation log file not found at {conversation_log_path}")
            
            st.session_state["page3_write_article_state"] = "final_writing"
            status.update(label="brain**STORM**ing complete!", state="complete")

def handle_final_writing():
    if st.session_state["page3_write_article_state"] == "final_writing":
        with st.status("Now I will connect the information I found for your reference. (This may take 4-5 minutes.)") as status:
            st.info('Now I will connect the information I found for your reference. (This may take 4-5 minutes.)')
            st.session_state["runner"].run(topic=st.session_state["page3_topic"], do_research=False,
                                           do_generate_outline=False,
                                           do_generate_article=True, do_polish_article=True, remove_duplicate=False)
            st.session_state["runner"].post_run()
            st.session_state["page3_write_article_state"] = "prepare_to_show_result"
            status.update(label="information synthesis complete!", state="complete")

def handle_prepare_to_show_result():
    if st.session_state["page3_write_article_state"] == "prepare_to_show_result":
        _, show_result_col, _ = st.columns([4, 3, 4])
        with show_result_col:
            if st.button("show final article"):
                st.session_state["page3_write_article_state"] = "completed"
                st.rerun()

def handle_completed():
    if st.session_state["page3_write_article_state"] == "completed":
        if st.session_state["use_costorm"]:
            st.header("Generated Co-STORM Article")
            st.markdown(st.session_state["costorm_article"])
            
            st.subheader("Final Mind Map")
            st.json(st.session_state["runner"].knowledge_base.to_dict())
            
            st.subheader("Conversation History")
            for turn in st.session_state["runner"].conversation_history:
                st.text(f"{turn['speaker']}: {turn['utterance']}")
        else:
            current_working_dir_paths = DemoFileIOHelper.read_structure_to_dict(
                st.session_state["page3_current_working_dir"])
            current_article_file_path_dict = current_working_dir_paths[st.session_state["page3_topic_name_truncated"]]
            demo_util.display_article_page(selected_article_name=st.session_state["page3_topic_name_cleaned"],
                                           selected_article_file_path_dict=current_article_file_path_dict,
                                           show_title=True, show_main_article=True)

def create_new_article_page():
    demo_util.clear_other_page_session_state(page_index=3)

    if "page3_write_article_state" not in st.session_state:
        st.session_state["page3_write_article_state"] = "not started"
    
    if "use_costorm" not in st.session_state:
        st.session_state["use_costorm"] = False
    
    if "page3_current_working_dir" not in st.session_state:
        st.session_state["page3_current_working_dir"] = os.path.join(demo_util.get_demo_dir(), "DEMO_WORKING_DIR")

    handle_not_started()
    handle_initiated()
    
    if st.session_state["use_costorm"]:
        handle_costorm_interaction()
    else:
        handle_pre_writing()
        handle_final_writing()
        handle_prepare_to_show_result()
    
    handle_completed()