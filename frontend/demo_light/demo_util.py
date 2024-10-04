# Standaard Python-modules voor bestandsverwerking, datummanipulatie en reguliere expressies
import base64
import datetime
import json
import os
import re
from typing import Optional

# Externe bibliotheken voor Markdown-verwerking, tijdzone-ondersteuning en Streamlit UI
import markdown
import pytz
import streamlit as st

# Configuratie voor lokale ontwikkeling
# Als je de broncode installeert in plaats van het 'knowledge-storm' pakket,
# haal de volgende regels uit commentaar:
import sys
sys.path.append('../../')

# Imports uit de knowledge-storm bibliotheek voor STORM Wiki functionaliteit
from knowledge_storm import STORMWikiRunnerArguments, STORMWikiRunner, STORMWikiLMConfigs
from knowledge_storm.lm import OllamaClient
from knowledge_storm.rm import SerperRM
from knowledge_storm.storm_wiki.modules.callback import BaseCallbackHandler
from knowledge_storm.utils import truncate_filename
from stoc import stoc
from knowledge_storm.collaborative_storm.engine import CollaborativeStormLMConfigs, RunnerArgument, CoStormRunner
from knowledge_storm.logging_wrapper import LoggingWrapper


class DemoFileIOHelper():
    """
    Hulpklasse voor bestandsverwerking en I/O-operaties in de demo-applicatie.
    Bevat statische methoden voor het lezen en verwerken van verschillende bestandstypen.
    """

    @staticmethod
    def read_structure_to_dict(articles_root_path):
        """
        Leest de mapstructuur van artikelen in het gegeven hoofdpad en
        retourneert een geneste dictionary. De buitenste dictionary heeft artikelnamen als sleutels,
        en elke waarde is een andere dictionary die bestandsnamen koppelt aan hun absolute paden.

        Args:
            articles_root_path (str): Het hoofdmappad met artikel-submappen.

        Returns:
            dict: Een dictionary waar elke sleutel een artikelnaam is, en elke waarde een dictionary
                van bestandsnamen en hun absolute paden binnen de map van dat artikel.
        """
        articles_dict = {}
        for topic_name in os.listdir(articles_root_path):
            topic_path = os.path.join(articles_root_path, topic_name)
            if os.path.isdir(topic_path):
                # Initialiseer of update de dictionary voor het onderwerp
                articles_dict[topic_name] = {}
                # Itereer over alle bestanden binnen een onderwerpsmap
                for file_name in os.listdir(topic_path):
                    file_path = os.path.join(topic_path, file_name)
                    articles_dict[topic_name][file_name] = os.path.abspath(file_path)
        return articles_dict

    @staticmethod
    def read_txt_file(file_path):
        """
        Leest de inhoud van een tekstbestand en retourneert het als een string.

        Args:
            file_path (str): Het pad naar het te lezen tekstbestand.

        Returns:
            str: De inhoud van het bestand als één string.
        """
        with open(file_path) as f:
            return f.read()

    @staticmethod
    def read_json_file(file_path):
        """
        Leest een JSON-bestand en retourneert de inhoud als een Python dictionary of lijst,
        afhankelijk van de JSON-structuur.

        Args:
            file_path (str): Het pad naar het te lezen JSON-bestand.

        Returns:
            dict or list: De inhoud van het JSON-bestand. Het type hangt af van de
                        structuur van het JSON-bestand (object of array op het hoogste niveau).
        """
        with open(file_path) as f:
            return json.load(f)

    @staticmethod
    def read_image_as_base64(image_path):
        """
        Leest een afbeeldingsbestand en retourneert de inhoud gecodeerd als een base64-string,
        geschikt voor insluiting in HTML of overdracht via netwerken waar binaire
        gegevens niet gemakkelijk kunnen worden verzonden.

        Args:
            image_path (str): Het pad naar het te coderen afbeeldingsbestand.

        Returns:
            str: De base64-gecodeerde string van de afbeelding, voorafgegaan door het benodigde
                data URI-schema voor afbeeldingen.
        """
        with open(image_path, "rb") as f:
            data = f.read()
            encoded = base64.b64encode(data)
        data = "data:image/png;base64," + encoded.decode("utf-8")
        return data

    @staticmethod
    def set_file_modification_time(file_path, modification_time_string):
        """
        Stelt de wijzigingstijd van een bestand in op basis van een gegeven tijdstring in de Californische tijdzone.

        Args:
            file_path (str): Het pad naar het bestand.
            modification_time_string (str): De gewenste wijzigingstijd in 'YYYY-MM-DD HH:MM:SS' formaat.
        """
        california_tz = pytz.timezone('America/Los_Angeles')
        modification_time = datetime.datetime.strptime(modification_time_string, '%Y-%m-%d %H:%M:%S')
        modification_time = california_tz.localize(modification_time)
        modification_time_utc = modification_time.astimezone(datetime.timezone.utc)
        modification_timestamp = modification_time_utc.timestamp()
        os.utime(file_path, (modification_timestamp, modification_timestamp))

    @staticmethod
    def get_latest_modification_time(path):
        """
        Retourneert de laatste wijzigingstijd van alle bestanden in een map in de Californische tijdzone als een string.

        Args:
            directory_path (str): Het pad naar de map.

        Returns:
            str: De laatste wijzigingstijd van het bestand in 'YYYY-MM-DD HH:MM:SS' formaat.
        """
        california_tz = pytz.timezone('America/Los_Angeles')
        latest_mod_time = None

        file_paths = []
        if os.path.isdir(path):
            for root, dirs, files in os.walk(path):
                for file in files:
                    file_paths.append(os.path.join(root, file))
        else:
            file_paths = [path]

        for file_path in file_paths:
            modification_timestamp = os.path.getmtime(file_path)
            modification_time_utc = datetime.datetime.utcfromtimestamp(modification_timestamp)
            modification_time_utc = modification_time_utc.replace(tzinfo=datetime.timezone.utc)
            modification_time_california = modification_time_utc.astimezone(california_tz)

            if latest_mod_time is None or modification_time_california > latest_mod_time:
                latest_mod_time = modification_time_california

        if latest_mod_time is not None:
            return latest_mod_time.strftime('%Y-%m-%d %H:%M:%S')
        else:
            return datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    @staticmethod
    def assemble_article_data(article_file_path_dict):
        """
        Construeert een dictionary met de inhoud en metadata van een artikel
        op basis van de beschikbare bestanden in de map van het artikel. Dit omvat de
        hoofdtekst van het artikel, citaten uit een JSON-bestand en een gesprekslogboek indien
        beschikbaar. De functie geeft voorrang aan een gepolijste versie van het artikel als
        zowel een ruwe als een gepolijste versie bestaan.

        Args:
            article_file_paths (dict): Een dictionary waar sleutels bestandsnamen zijn die relevant
                                    zijn voor het artikel (bijv. de artikeltekst, citaten
                                    in JSON-formaat, gesprekslogboeken) en waarden
                                    hun corresponderende bestandspaden zijn.

        Returns:
            dict of None: Een dictionary met de geparseerde inhoud van het artikel,
                        citaten en gesprekslogboek indien beschikbaar. Retourneert None
                        als noch de ruwe noch de gepolijste artikeltekst bestaat in de
                        opgegeven bestandspaden.
        """
        if "storm_gen_article.txt" in article_file_path_dict or "storm_gen_article_polished.txt" in article_file_path_dict:
            full_article_name = "storm_gen_article_polished.txt" if "storm_gen_article_polished.txt" in article_file_path_dict else "storm_gen_article.txt"
            article_data = {"article": DemoTextProcessingHelper.parse(
                DemoFileIOHelper.read_txt_file(article_file_path_dict[full_article_name]))}
            if "url_to_info.json" in article_file_path_dict:
                article_data["citations"] = _construct_citation_dict_from_search_result(
                    DemoFileIOHelper.read_json_file(article_file_path_dict["url_to_info.json"]))
            if "conversation_log.json" in article_file_path_dict:
                article_data["conversation_log"] = DemoFileIOHelper.read_json_file(
                    article_file_path_dict["conversation_log.json"])
            return article_data
        return None


class DemoTextProcessingHelper():
    """
    Hulpklasse voor tekstverwerking in de demo-applicatie.
    Bevat statische methoden voor het parseren en formatteren van tekst.
    """

    @staticmethod
    def remove_citations(sent):
        """
        Verwijdert citaatmarkeringen uit een zin.

        Args:
            sent (str): De te verwerken zin.

        Returns:
            str: De zin zonder citaatmarkeringen.
        """
        return re.sub(r"\[\d+", "", re.sub(r" \[\d+", "", sent)).replace(" |", "").replace("]", "")

    @staticmethod
    def parse_conversation_history(json_data):
        """
        Parseert conversatiegeschiedenis uit JSON-data.

        Args:
            json_data (list): Lijst met conversatiegegevens in JSON-formaat.

        Returns:
            list: Lijst van tuples met geparste conversatiegegevens (naam, beschrijving, dialoogbeurten).
        """
        parsed_data = []
        for persona_conversation_data in json_data:
            if ': ' in persona_conversation_data["perspective"]:
                name, description = persona_conversation_data["perspective"].split(": ", 1)
            elif '- ' in persona_conversation_data["perspective"]:
                name, description = persona_conversation_data["perspective"].split("- ", 1)
            else:
                name, description = "", persona_conversation_data["perspective"]
            cur_conversation = []
            for dialogue_turn in persona_conversation_data["dlg_turns"]:
                cur_conversation.append({"role": "user", "content": dialogue_turn["user_utterance"]})
                cur_conversation.append(
                    {"role": "assistant",
                     "content": DemoTextProcessingHelper.remove_citations(dialogue_turn["agent_utterance"])})
            parsed_data.append((name, description, cur_conversation))
        return parsed_data

    @staticmethod
    def parse(text):
        """
        Verwijdert specifieke patronen uit de tekst.

        Args:
            text (str): De te verwerken tekst.

        Returns:
            str: De verwerkte tekst.
        """
        regex = re.compile(r']:\s+"(.*?)"\s+http')
        text = regex.sub(']: http', text)
        return text

    @staticmethod
    def add_markdown_indentation(input_string):
        """
        Voegt inspringing toe aan Markdown-tekst op basis van het aantal hekjes (#).

        Args:
            input_string (str): De te verwerken Markdown-tekst.

        Returns:
            str: De ingesproken Markdown-tekst.
        """
        lines = input_string.split('\n')
        processed_lines = [""]
        for line in lines:
            num_hashes = 0
            for char in line:
                if char == '#':
                    num_hashes += 1
                else:
                    break
            num_spaces = 4 * num_hashes
            new_line = ' ' * num_spaces + line
            processed_lines.append(new_line)
        return '\n'.join(processed_lines)

    @staticmethod
    def get_current_time_string():
        """
        Geeft de huidige tijd in de Californische tijdzone als string.

        Returns:
            str: De huidige Californische tijd in 'YYYY-MM-DD HH:MM:SS' formaat.
        """
        california_tz = pytz.timezone('America/Los_Angeles')
        utc_now = datetime.datetime.now(datetime.timezone.utc)
        california_now = utc_now.astimezone(california_tz)
        return california_now.strftime('%Y-%m-%d %H:%M:%S')

    @staticmethod
    def compare_time_strings(time_string1, time_string2, time_format='%Y-%m-%d %H:%M:%S'):
        """
        Vergelijkt twee tijdstrings om te bepalen of ze hetzelfde tijdstip vertegenwoordigen.

        Args:
            time_string1 (str): De eerste tijdstring.
            time_string2 (str): De tweede tijdstring.
            time_format (str): Het formaat van de tijdstrings.

        Returns:
            bool: True als de tijdstrings hetzelfde tijdstip vertegenwoordigen, anders False.
        """
        time1 = datetime.datetime.strptime(time_string1, time_format)
        time2 = datetime.datetime.strptime(time_string2, time_format)
        return time1 == time2

    @staticmethod
    def add_inline_citation_link(article_text, citation_dict):
        """
        Voegt inline citatielinks toe aan de artikeltekst.

        Args:
            article_text (str): De artikeltekst.
            citation_dict (dict): Dictionary met citatie-informatie.

        Returns:
            str: De artikeltekst met toegevoegde inline citatielinks.
        """
        pattern = r'\[(\d+)\]'

        def replace_with_link(match):
            i = match.group(1)
            url = citation_dict.get(int(i), {}).get('url', '#')
            return f'[[{i}]]({url})'

        return re.sub(pattern, replace_with_link, article_text)

    @staticmethod
    def generate_html_toc(md_text):
        """
        Genereert een HTML-inhoudsopgave op basis van Markdown-koppen.

        Args:
            md_text (str): De Markdown-tekst.

        Returns:
            str: HTML-string met de inhoudsopgave.
        """
        toc = []
        for line in md_text.splitlines():
            if line.startswith("#"):
                level = line.count("#")
                title = line.strip("# ").strip()
                anchor = title.lower().replace(" ", "-").replace(".", "")
                toc.append(f"<li style='margin-left: {20 * (level - 1)}px;'><a href='#{anchor}'>{title}</a></li>")
        return "<ul>" + "".join(toc) + "</ul>"

    @staticmethod
    def construct_bibliography_from_url_to_info(url_to_info):
        """
        Construeert een bibliografie op basis van URL-informatie.

        Args:
            url_to_info (dict): Dictionary met URL-informatie.

        Returns:
            str: Geformatteerde bibliografie als Markdown-tekst.
        """
        bibliography_list = []
        sorted_url_to_unified_index = dict(sorted(url_to_info['url_to_unified_index'].items(),
                                                  key=lambda item: item[1]))
        for url, index in sorted_url_to_unified_index.items():
            title = url_to_info['url_to_info'][url]['title']
            bibliography_list.append(f"[{index}]: [{title}]({url})")
        bibliography_string = "\n\n".join(bibliography_list)
        return f"# References\n\n{bibliography_string}"


class DemoUIHelper():
    """
    Hulpklasse voor UI-gerelateerde functies in de demo-applicatie.
    Bevat statische methoden voor het aanpassen van de Streamlit UI.
    """

    @staticmethod
    def st_markdown_adjust_size(content, font_size=20):
        """
        Past de lettergrootte aan van Markdown-inhoud in Streamlit.

        Args:
            content (str): De weer te geven Markdown-inhoud.
            font_size (int): De gewenste lettergrootte in pixels.
        """
        st.markdown(f"""
        <span style='font-size: {font_size}px;'>{content}</span>
        """, unsafe_allow_html=True)

    @staticmethod
    def get_article_card_UI_style(boarder_color="#9AD8E1"):
        """
        Definieert de UI-stijl voor een artikelkaart.

        Args:
            boarder_color (str): De kleur van de rand van de kaart.

        Returns:
            dict: Een dictionary met CSS-stijlen voor verschillende elementen van de kaart.
        """
        return {
            "card": {
                "width": "100%",
                "height": "116px",
                "max-width": "640px",
                "background-color": "#FFFFF",
                "border": "1px solid #CCC",
                "padding": "20px",
                "border-radius": "5px",
                "border-left": f"0.5rem solid {boarder_color}",
                "box-shadow": "0 0.15rem 1.75rem 0 rgba(58, 59, 69, 0.15)",
                "margin": "0px"
            },
            "title": {
                "white-space": "nowrap",
                "overflow": "hidden",
                "text-overflow": "ellipsis",
                "font-size": "17px",
                "color": "rgb(49, 51, 63)",
                "text-align": "left",
                "width": "95%",
                "font-weight": "normal"
            },
            "text": {
                "white-space": "nowrap",
                "overflow": "hidden",
                "text-overflow": "ellipsis",
                "font-size": "25px",
                "color": "rgb(49, 51, 63)",
                "text-align": "left",
                "width": "95%"
            },
            "filter": {
                "background-color": "rgba(0, 0, 0, 0)"
            }
        }

    @staticmethod
    def customize_toast_css_style():
        """
        Past de CSS-stijl aan voor toast-meldingen in Streamlit.
        """
        st.markdown(
            """
            <style>
                div[data-testid=stToast] {
                    padding: 20px 10px 40px 10px;
                    background-color: #FF0000;   /* red */
                    width: 40%;
                }

                [data-testid=toastContainer] [data-testid=stMarkdownContainer] > p {
                    font-size: 25px;
                    font-style: normal;
                    font-weight: 400;
                    color: #FFFFFF;   /* white */
                    line-height: 1.5; /* Adjust this value as needed */
                }
            </style>
            """, unsafe_allow_html=True
        )

    @staticmethod
    def article_markdown_to_html(article_title, article_content):
        """
        Zet een Markdown-artikel om naar HTML met een inhoudsopgave.

        Args:
            article_title (str): De titel van het artikel.
            article_content (str): De inhoud van het artikel in Markdown-formaat.

        Returns:
            str: HTML-representatie van het artikel met inhoudsopgave.
        """
        return f"""
        <html>
            <head>
                <meta charset="utf-8">
                <title>{article_title}</title>
                <style>
                    .title {{
                        text-align: center;
                    }}
                </style>
            </head>
            <body>
                <div class="title">
                    <h1>{article_title.replace('_', ' ')}</h1>
                </div>
                <h2>Table of Contents</h2>
                {DemoTextProcessingHelper.generate_html_toc(article_content)}
                {markdown.markdown(article_content)}
            </body>
        </html>
        """


def _construct_citation_dict_from_search_result(search_results):
    """
    Construeert een citatie-dictionary op basis van zoekresultaten.

    Args:
        search_results (dict): Dictionary met zoekresultaten.

    Returns:
        dict: Een dictionary met citatie-informatie.
    """
    if search_results is None:
        return None
    citation_dict = {}
    for url, index in search_results['url_to_unified_index'].items():
        citation_dict[index] = {'url': url,
                                'title': search_results['url_to_info'][url]['title'],
                                'snippets': search_results['url_to_info'][url]['snippets']}
    return citation_dict


def _display_main_article_text(article_text, citation_dict, table_content_sidebar):
    """
    Toont de hoofdtekst van het artikel met citaties en een inhoudsopgave.

    Args:
        article_text (str): De tekst van het artikel.
        citation_dict (dict): Dictionary met citatie-informatie.
        table_content_sidebar: Streamlit-container voor de inhoudsopgave.
    """
    if "Write the lead section:" in article_text:
        article_text = article_text[
                       article_text.find("Write the lead section:") + len("Write the lead section:"):]
    if article_text[0] == '#':
        article_text = '\n'.join(article_text.split('\n')[1:])
    article_text = DemoTextProcessingHelper.add_inline_citation_link(article_text, citation_dict)
    article_text = article_text.replace("$", "\\$")
    stoc.from_markdown(article_text, table_content_sidebar)


def _display_references(citation_dict):
    """
    Toont referenties in een Streamlit-interface.

    Args:
        citation_dict (dict): Dictionary met citatie-informatie.
    """
    if citation_dict:
        reference_list = [f"reference [{i}]" for i in range(1, len(citation_dict) + 1)]
        selected_key = st.selectbox("Select a reference", reference_list)
        citation_val = citation_dict[reference_list.index(selected_key) + 1]
        citation_val['title'] = citation_val['title'].replace("$", "\\$")
        st.markdown(f"**Title:** {citation_val['title']}")
        st.markdown(f"**Url:** {citation_val['url']}")
        snippets = '\n\n'.join(citation_val['snippets']).replace("$", "\\$")
        st.markdown(f"**Highlights:**\n\n {snippets}")
    else:
        st.markdown("**No references available**")


def _display_persona_conversations(conversation_log):
    """
    Toont persona-gesprekken in een dialoog-UI.

    Args:
        conversation_log (list): Lijst met conversatiegegevens.
    """
    parsed_conversation_history = DemoTextProcessingHelper.parse_conversation_history(conversation_log)
    persona_tabs = st.tabs([name for (name, _, _) in parsed_conversation_history])
    for idx, persona_tab in enumerate(persona_tabs):
        with persona_tab:
            st.info(parsed_conversation_history[idx][1])
            for message in parsed_conversation_history[idx][2]:
                message['content'] = message['content'].replace("$", "\\$")
                with st.chat_message(message["role"]):
                    if message["role"] == "user":
                        st.markdown(f"**{message['content']}**")
                    else:
                        st.markdown(message["content"])


def _display_main_article(selected_article_file_path_dict, show_reference=True, show_conversation=True):
    """
    Toont het hoofdartikel met referenties en conversatiegeschiedenis.

    Args:
        selected_article_file_path_dict (dict): Dictionary met bestandspaden voor het geselecteerde artikel.
        show_reference (bool): Of referenties moeten worden getoond.
        show_conversation (bool): Of de conversatiegeschiedenis moet worden getoond.
    """
    article_data = DemoFileIOHelper.assemble_article_data(selected_article_file_path_dict)

    with st.container(height=1000, border=True):
        table_content_sidebar = st.sidebar.expander("**Table of contents**", expanded=True)
        _display_main_article_text(article_text=article_data.get("article", ""),
                                   citation_dict=article_data.get("citations", {}),
                                   table_content_sidebar=table_content_sidebar)

    if show_reference and "citations" in article_data:
        with st.sidebar.expander("**References**", expanded=True):
            with st.container(height=800, border=False):
                _display_references(citation_dict=article_data.get("citations", {}))

    if show_conversation and "conversation_log" in article_data:
        with st.expander(
                "**STORM** is powered by a knowledge agent that proactively research a given topic by asking good questions coming from different perspectives.\n\n"
                ":sunglasses: Click here to view the agent's brain**STORM**ing process!"):
            _display_persona_conversations(conversation_log=article_data.get("conversation_log", {}))


def get_demo_dir():
    """
    Bepaalt de directory van het demo-script.

    Returns:
        str: Absoluut pad naar de demo-directory.
    """
    return os.path.dirname(os.path.abspath(__file__))


def clear_other_page_session_state(page_index: Optional[int]):
    """
    Wist sessiestatussen van andere pagina's.

    Args:
        page_index (Optional[int]): Index van de huidige pagina. Als None, worden alle paginastatussen gewist.
    """
    if page_index is None:
        keys_to_delete = [key for key in st.session_state if key.startswith("page")]
    else:
        keys_to_delete = [key for key in st.session_state if key.startswith("page") and f"page{page_index}" not in key]
    for key in set(keys_to_delete):
        del st.session_state[key]


def set_storm_runner():
    llm_configs, rm = initialize_llm_and_rm()
    
    current_working_dir = os.path.join(get_demo_dir(), "DEMO_WORKING_DIR")
    if not os.path.exists(current_working_dir):
        os.makedirs(current_working_dir)
    
    engine_args = STORMWikiRunnerArguments(
        output_dir=current_working_dir,
        max_conv_turn=3,
        max_perspective=3,
        search_top_k=3,
        retrieve_top_k=5
    )
    
    if st.session_state.get("use_costorm", False):
        st.session_state["runner"] = initialize_costorm_runner(llm_configs, st.session_state["page3_topic"], rm)
    else:
        st.session_state["runner"] = STORMWikiRunner(engine_args, llm_configs, rm)


def display_article_page(selected_article_name, selected_article_file_path_dict,
                         show_title=True, show_main_article=True):
    """
    Toont een artikelpagina in de Streamlit-interface.

    Args:
        selected_article_name (str): Naam van het geselecteerde artikel.
        selected_article_file_path_dict (dict): Dictionary met bestandspaden voor het geselecteerde artikel.
        show_title (bool): Of de titel moet worden getoond.
        show_main_article (bool): Of het hoofdartikel moet worden getoond.
    """
    if show_title:
        st.markdown(f"<h2 style='text-align: center;'>{selected_article_name.replace('_', ' ')}</h2>",
                    unsafe_allow_html=True)

    if show_main_article:
        _display_main_article(selected_article_file_path_dict)


class StreamlitCallbackHandler(BaseCallbackHandler):
    """
    Callback-handler voor Streamlit-updates tijdens het STORM-proces.
    """

    def __init__(self, status_container):
        """
        Initialiseert de StreamlitCallbackHandler.

        Args:
            status_container: Streamlit-container voor statusupdates.
        """
        self.status_container = status_container

    def on_identify_perspective_start(self, **kwargs):
        """
        Callback bij het starten van perspectiefidentificatie.
        """
        self.status_container.info('Start identifying different perspectives for researching the topic.')

    def on_identify_perspective_end(self, perspectives: list[str], **kwargs):
        """
        Callback bij het voltooien van perspectiefidentificatie.

        Args:
            perspectives (list[str]): Lijst van geïdentificeerde perspectieven.
        """
        perspective_list = "\n- ".join(perspectives)
        self.status_container.success(f'Finish identifying perspectives. Will now start gathering information'
                                      f' from the following perspectives:\n- {perspective_list}')

    def on_information_gathering_start(self, **kwargs):
        """
        Callback bij het starten van informatieverzameling.
        """
        self.status_container.info('Start browsing the Internet.')

    def on_dialogue_turn_end(self, dlg_turn, **kwargs):
        """
        Callback bij het voltooien van een dialoogbeurt.

        Args:
            dlg_turn: Object met informatie over de voltooide dialoogbeurt.
        """
        urls = list(set([r.url for r in dlg_turn.search_results]))
        for url in urls:
            self.status_container.markdown(f"""
                    <style>
                    .small-font {{
                        font-size: 14px;
                        margin: 0px;
                        padding: 0px;
                    }}
                    </style>
                    <div class="small-font">Finish browsing <a href="{url}" class="small-font" target="_blank">{url}</a>.</div>
                    """, unsafe_allow_html=True)

    def on_information_gathering_end(self, **kwargs):
        """
        Callback bij het voltooien van informatieverzameling.
        """
        self.status_container.success('Finish collecting information.')

    def on_information_organization_start(self, **kwargs):
        """
        Callback bij het starten van informatie-organisatie.
        """
        self.status_container.info('Start organizing information into a hierarchical outline.')

    def on_direct_outline_generation_end(self, outline: str, **kwargs):
        """
        Callback bij het voltooien van directe outline-generatie.

        Args:
            outline (str): De gegenereerde outline.
        """
        self.status_container.success(f'Finish leveraging the internal knowledge of the large language model.')

    def on_outline_refinement_end(self, outline: str, **kwargs):
        """
        Callback bij het voltooien van outline-verfijning.

        Args:
            outline (str): De verfijnde outline.
        """
        self.status_container.success(f'Finish leveraging the collected information.')

def initialize_llm_and_rm():
    llm_configs = STORMWikiLMConfigs()
    
    # Initialiseer Ollama modellen
    llm_configs.init_ollama_model(
        model='hermes3:8b-llama3.1-fp16', 
        port=11434,
        url="http://localhost",
        max_tokens=128000, 
        temperature=1.0, 
        top_p=0.9
    )
    
    engine_args = STORMWikiRunnerArguments(
        output_dir="./results",
        max_conv_turn=3,
        max_perspective=3,
        search_top_k=3,
        retrieve_top_k=5
    )

    rm = SerperRM(serper_search_api_key=st.secrets['SERPER_API_KEY'], k=engine_args.search_top_k)

    return llm_configs, rm

def initialize_costorm_runner(llm_configs, topic, rm):
    lm_config = CollaborativeStormLMConfigs()
    
    lm_config.set_question_answering_lm(llm_configs.article_gen_lm)
    lm_config.set_discourse_manage_lm(llm_configs.conv_simulator_lm)
    lm_config.set_utterance_polishing_lm(llm_configs.article_polish_lm)
    lm_config.set_warmstart_outline_gen_lm(llm_configs.outline_gen_lm)
    lm_config.set_question_asking_lm(llm_configs.question_asker_lm)
    lm_config.set_knowledge_base_lm(llm_configs.article_gen_lm)

    runner_argument = RunnerArgument(topic=topic)
    logging_wrapper = LoggingWrapper(lm_config)
    
    return CoStormRunner(lm_config=lm_config,
                         runner_argument=runner_argument,
                         logging_wrapper=logging_wrapper,
                         rm=rm)

def run_storm(topic, use_costorm=False):
    llm_configs, rm = initialize_llm_and_rm()
    
    if use_costorm:
        runner = initialize_costorm_runner(llm_configs, topic, rm)
        runner.warm_start()
        return runner
    else:
        # Bestaande STORM logica
        engine_args = STORMWikiRunnerArguments(
            output_dir="./results",
            max_conv_turn=3,
            max_perspective=3,
            search_top_k=3,
            max_thread_num=3,
        )
        runner = STORMWikiRunner(engine_args, llm_configs, rm)
        runner.run(
            topic=topic,
            do_research=True,
            do_generate_outline=True,
            do_generate_article=True,
            do_polish_article=True,
        )
        return runner.get_result()