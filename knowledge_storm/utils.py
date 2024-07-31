import concurrent.futures
import json
import logging
import os
import pickle
import re
import sys
from typing import List, Dict

import httpx
import toml
from langchain_text_splitters import RecursiveCharacterTextSplitter
from trafilatura import extract

# Configureer logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

logging.getLogger("httpx").setLevel(logging.WARNING)  # Disable INFO logging for httpx.

def load_api_key(toml_file_path):
    logger.info(f"Attempting to load API key from {toml_file_path}")
    try:
        with open(toml_file_path, 'r') as file:
            data = toml.load(file)
        logger.debug("TOML file loaded successfully")
    except FileNotFoundError:
        logger.error(f"File not found: {toml_file_path}")
        return
    except toml.TomlDecodeError:
        logger.error(f"Error decoding TOML file: {toml_file_path}")
        return
    # Set environment variables
    for key, value in data.items():
        os.environ[key] = str(value)
        logger.debug(f"Environment variable set: {key}")
    logger.info("API key loaded and environment variables set")

def makeStringRed(message):
    return f"\033[91m {message}\033[00m"

class ArticleTextProcessing:
    @staticmethod
    def limit_word_count_preserve_newline(input_string, max_word_count):
        logger.debug(f"Limiting word count to {max_word_count}")
        word_count = 0
        limited_string = ''

        for word in input_string.split('\n'):
            line_words = word.split()
            for lw in line_words:
                if word_count < max_word_count:
                    limited_string += lw + ' '
                    word_count += 1
                else:
                    break
            if word_count >= max_word_count:
                break
            limited_string = limited_string.strip() + '\n'

        logger.debug(f"Limited string to {word_count} words")
        return limited_string.strip()

    @staticmethod
    def remove_citations(s):
        logger.debug("Removing citations from string")
        result = re.sub(r'\[\d+(?:,\s*\d+)*\]', '', s)
        logger.debug("Citations removed")
        return result

    @staticmethod
    def parse_citation_indices(s):
        logger.debug("Parsing citation indices")
        matches = re.findall(r'\[\d+\]', s)
        indices = [int(index[1:-1]) for index in matches]
        logger.debug(f"Found {len(indices)} citation indices")
        return indices

    @staticmethod
    def remove_uncompleted_sentences_with_citations(text):
        logger.debug("Removing uncompleted sentences and standalone citations")
        
        def replace_with_individual_brackets(match):
            numbers = match.group(1).split(', ')
            return ' '.join(f'[{n}]' for n in numbers)

        def deduplicate_group(match):
            citations = match.group(0)
            unique_citations = list(set(re.findall(r'\[\d+\]', citations)))
            sorted_citations = sorted(unique_citations, key=lambda x: int(x.strip('[]')))
            return ''.join(sorted_citations)

        text = re.sub(r'\[([0-9, ]+)\]', replace_with_individual_brackets, text)
        text = re.sub(r'(\[\d+\])+', deduplicate_group, text)

        eos_pattern = r'([.!?])\s*(\[\d+\])?\s*'
        matches = list(re.finditer(eos_pattern, text))
        if matches:
            last_match = matches[-1]
            text = text[:last_match.end()].strip()

        logger.debug("Uncompleted sentences and standalone citations removed")
        return text

    @staticmethod
    def clean_up_citation(conv):
        logger.info("Cleaning up citations in conversation")
        for turn in conv.dlg_history:
            turn.agent_utterance = turn.agent_utterance[:turn.agent_utterance.find('References:')]
            turn.agent_utterance = turn.agent_utterance[:turn.agent_utterance.find('Sources:')]
            turn.agent_utterance = turn.agent_utterance.replace('Answer:', '').strip()
            try:
                max_ref_num = max([int(x) for x in re.findall(r'\[(\d+)\]', turn.agent_utterance)])
            except Exception as e:
                logger.warning(f"Error finding max reference number: {e}")
                max_ref_num = 0
            if max_ref_num > len(turn.search_results):
                for i in range(len(turn.search_results), max_ref_num + 1):
                    turn.agent_utterance = turn.agent_utterance.replace(f'[{i}]', '')
            turn.agent_utterance = ArticleTextProcessing.remove_uncompleted_sentences_with_citations(
                turn.agent_utterance)
        logger.info("Citation cleanup completed")
        return conv

    @staticmethod
    def clean_up_outline(outline, topic=""):
        logger.info(f"Cleaning up outline for topic: {topic}")
        output_lines = []
        current_level = 0  # To track the current section level

        for line in outline.split('\n'):
            stripped_line = line.strip()

            if topic != "" and f"# {topic.lower()}" in stripped_line.lower():
                output_lines = []

            if stripped_line.startswith('#'):
                current_level = stripped_line.count('#')
                output_lines.append(stripped_line)
            elif stripped_line.startswith('-'):
                subsection_header = '#' * (current_level + 1) + ' ' + stripped_line[1:].strip()
                output_lines.append(subsection_header)

        outline = '\n'.join(output_lines)

        # Remove references.
        patterns_to_remove = [
            r"#[#]? See also.*?(?=##|$)",
            r"#[#]? See Also.*?(?=##|$)",
            r"#[#]? Notes.*?(?=##|$)",
            r"#[#]? References.*?(?=##|$)",
            r"#[#]? External links.*?(?=##|$)",
            r"#[#]? External Links.*?(?=##|$)",
            r"#[#]? Bibliography.*?(?=##|$)",
            r"#[#]? Further reading*?(?=##|$)",
            r"#[#]? Further Reading*?(?=##|$)",
            r"#[#]? Summary.*?(?=##|$)",
            r"#[#]? Appendices.*?(?=##|$)",
            r"#[#]? Appendix.*?(?=##|$)",
        ]

        for pattern in patterns_to_remove:
            outline = re.sub(pattern, '', outline, flags=re.DOTALL)
            logger.debug(f"Removed section matching pattern: {pattern}")

        logger.info("Outline cleanup completed")
        return outline

    @staticmethod
    def clean_up_section(text):
        logger.info("Cleaning up section")
        paragraphs = text.split('\n')
        output_paragraphs = []
        summary_sec_flag = False
        for p in paragraphs:
            p = p.strip()
            if len(p) == 0:
                continue
            if not p.startswith('#'):
                p = ArticleTextProcessing.remove_uncompleted_sentences_with_citations(p)
            if summary_sec_flag:
                if p.startswith('#'):
                    summary_sec_flag = False
                else:
                    continue
            if p.startswith('Overall') or p.startswith('In summary') or p.startswith('In conclusion'):
                logger.debug("Skipping summary-like paragraph")
                continue
            if "# Summary" in p or '# Conclusion' in p:
                summary_sec_flag = True
                logger.debug("Summary section detected")
                continue
            output_paragraphs.append(p)

        logger.info("Section cleanup completed")
        return '\n\n'.join(output_paragraphs)

    @staticmethod
    def update_citation_index(s, citation_map):
        logger.info("Updating citation indices")
        for original_citation in citation_map:
            s = s.replace(f"[{original_citation}]", f"__PLACEHOLDER_{original_citation}__")
        for original_citation, unify_citation in citation_map.items():
            s = s.replace(f"__PLACEHOLDER_{original_citation}__", f"[{unify_citation}]")
        logger.debug(f"Updated {len(citation_map)} citation indices")
        return s

    @staticmethod
    def parse_article_into_dict(input_string):
        logger.info("Parsing article into dictionary structure")
        lines = input_string.split('\n')
        lines = [line for line in lines if line.strip()]
        root = {'content': '', 'subsections': {}}
        current_path = [(root, -1)]  # (current_dict, level)

        for line in lines:
            if line.startswith('#'):
                level = line.count('#')
                title = line.strip('# ').strip()
                new_section = {'content': '', 'subsections': {}}

                while current_path and current_path[-1][1] >= level:
                    current_path.pop()

                current_path[-1][0]['subsections'][title] = new_section
                current_path.append((new_section, level))
                logger.debug(f"Added new section: {title} at level {level}")
            else:
                current_path[-1][0]['content'] += line + '\n'

        logger.info(f"Parsed article into {len(root['subsections'])} top-level sections")
        return root['subsections']

class FileIOHelper:
    @staticmethod
    def dump_json(obj, file_name, encoding="utf-8"):
        logger.info(f"Dumping JSON to file: {file_name}")
        with open(file_name, 'w', encoding=encoding) as fw:
            json.dump(obj, fw, default=FileIOHelper.handle_non_serializable)
        logger.debug("JSON dump completed")

    @staticmethod
    def handle_non_serializable(obj):
        logger.warning(f"Encountered non-serializable object: {type(obj)}")
        return "non-serializable contents"

    @staticmethod
    def load_json(file_name, encoding="utf-8"):
        logger.info(f"Loading JSON from file: {file_name}")
        with open(file_name, 'r', encoding=encoding) as fr:
            data = json.load(fr)
        logger.debug("JSON load completed")
        return data

    @staticmethod
    def write_str(s, path):
        logger.info(f"Writing string to file: {path}")
        with open(path, 'w') as f:
            f.write(s)
        logger.debug("String write completed")

    @staticmethod
    def load_str(path):
        logger.info(f"Loading string from file: {path}")
        with open(path, 'r') as f:
            content = '\n'.join(f.readlines())
        logger.debug("String load completed")
        return content

    @staticmethod
    def dump_pickle(obj, path):
        logger.info(f"Dumping pickle to file: {path}")
        with open(path, 'wb') as f:
            pickle.dump(obj, f)
        logger.debug("Pickle dump completed")

    @staticmethod
    def load_pickle(path):
        logger.info(f"Loading pickle from file: {path}")
        with open(path, 'rb') as f:
            data = pickle.load(f)
        logger.debug("Pickle load completed")
        return data

class WebPageHelper:
    def __init__(self, min_char_count: int = 150, snippet_chunk_size: int = 1000, max_thread_num: int = 10):
        logger.info(f"Initializing WebPageHelper with min_char_count={min_char_count}, snippet_chunk_size={snippet_chunk_size}, max_thread_num={max_thread_num}")
        self.httpx_client = httpx.Client(verify=False)
        self.min_char_count = min_char_count
        self.max_thread_num = max_thread_num
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=snippet_chunk_size,
            chunk_overlap=0,
            length_function=len,
            is_separator_regex=False,
            separators=[
                "\n\n", "\n", ".", "\uff0e", "\u3002", ",", "\uff0c", "\u3001", " ", "\u200B", "",
            ],
        )
        logger.debug("WebPageHelper initialized")

    def download_webpage(self, url: str):
        logger.info(f"Downloading webpage: {url}")
        try:
            res = self.httpx_client.get(url, timeout=4)
            if res.status_code >= 400:
                res.raise_for_status()
            logger.debug(f"Successfully downloaded webpage: {url}")
            return res.content
        except httpx.HTTPError as exc:
            logger.error(f"Error while requesting {exc.request.url!r} - {exc!r}")
            return None

    def urls_to_articles(self, urls: List[str]) -> Dict:
        logger.info(f"Converting {len(urls)} URLs to articles")
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_thread_num) as executor:
            htmls = list(executor.map(self.download_webpage, urls))

        articles = {}

        for h, u in zip(htmls, urls):
            if h is None:
                logger.warning(f"Failed to download content for URL: {u}")
                continue
            article_text = extract(
                h,
                include_tables=False,
                include_comments=False,
                output_format="text",
            )
            if article_text is not None and len(article_text) > self.min_char_count:
                articles[u] = {"text": article_text}
                logger.debug(f"Extracted article from URL: {u}")
            else:
                logger.warning(f"Article text too short or None for URL: {u}")

        logger.info(f"Converted {len(articles)} URLs to articles")
        return articles

    def urls_to_snippets(self, urls: List[str]) -> Dict:
        logger.info(f"Converting {len(urls)} URLs to snippets")
        articles = self.urls_to_articles(urls)
        for u in articles:
            articles[u]["snippets"] = self.text_splitter.split_text(articles[u]["text"])
            logger.debug(f"Created {len(articles[u]['snippets'])} snippets for URL: {u}")

        logger.info(f"Converted {len(articles)} URLs to snippets")
        return articles

# Voeg een handler toe om logs naar een bestand te schrijven
file_handler = logging.FileHandler('web_processing.log')
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
logger.addHandler(file_handler)

logger.info("Web processing module initialized and logging configured")