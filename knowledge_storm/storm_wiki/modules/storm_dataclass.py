import copy
import re
import logging
from collections import OrderedDict
from typing import Union, Optional, Any, List, Tuple, Dict

import numpy as np

from sklearn.metrics.pairwise import cosine_similarity
from langchain_ollama import OllamaEmbeddings
from ...interface import Information, InformationTable, Article, ArticleSectionNode
from ...utils import ArticleTextProcessing, FileIOHelper

import streamlit as st

OLLAMA_MODEL_NAME = st.secrets["OLLAMA_MODEL_NAME"]
EMBEDDING_MODEL_NAME = st.secrets["EMBEDDING_MODEL_NAME"]

# Configureer logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class StormInformation(Information):
    def __init__(self, uuid=None, description=None, snippets=None, title=None, url=None, query=None, answer=None):
        super().__init__(uuid=uuid or url or '', meta={})
        self.description = description or ''
        self.snippets = snippets or []
        self.title = title or ''
        self.url = url or uuid or ''
        self.query = query
        self.answer = answer
        logger.debug(f"StormInformation object created with UUID: {self.uuid}")

    @classmethod
    def from_dict(cls, info_dict):
        uuid = info_dict.get('uuid') or info_dict.get('url') or ''
        logger.debug(f"Creating StormInformation from dict with UUID/URL: {uuid}")
        return cls(
            uuid=uuid,
            description=info_dict.get('description', ''),
            snippets=info_dict.get('snippets', []),
            title=info_dict.get('title', ''),
            url=info_dict.get('url', uuid),
            query=info_dict.get('query'),
            answer=info_dict.get('answer')
        )

    def to_dict(self):
        logger.debug(f"Converting StormInformation to dict: {self.uuid}")
        return {
            "uuid": self.uuid,
            "url": self.url,
            "description": self.description,
            "snippets": self.snippets,
            "title": self.title,
            "query": self.query,
            "answer": self.answer
        }

class DialogueTurn:
    def __init__(self, agent_utterance: str = None, user_utterance: str = None,
                 search_queries: Optional[List[str]] = None,
                 search_results: Optional[List[Union[StormInformation, Dict]]] = None):
        self.agent_utterance = agent_utterance
        self.user_utterance = user_utterance
        self.search_queries = search_queries
        self.search_results = search_results

        if self.search_results:
            for idx in range(len(self.search_results)):
                if type(self.search_results[idx]) == dict:
                    self.search_results[idx] = StormInformation.from_dict(self.search_results[idx])
        logger.debug(f"DialogueTurn created with {len(self.search_results) if self.search_results else 0} search results")

    def log(self):
        logger.debug("Logging DialogueTurn")
        return OrderedDict({
            'agent_utterance': self.agent_utterance,
            'user_utterance': self.user_utterance,
            'search_queries': self.search_queries,
            'search_results': [data.to_dict() for data in self.search_results],
        })

class StormInformationTable(InformationTable):
    def __init__(self, conversations=List[Tuple[str, List[DialogueTurn]]], embedding_model_name=None):
        super().__init__()
        self.conversations = conversations
        self.url_to_info: Dict[str, StormInformation] = self.construct_url_to_info(self.conversations)
        self.encoder = None
        self.collected_urls = []
        self.collected_snippets = []
        self.encoded_snippets = None
        self.embedding_model_name = embedding_model_name or EMBEDDING_MODEL_NAME
        logger.info(f"StormInformationTable initialized with {len(self.conversations)} conversations and embedding model: {self.embedding_model_name}")

    @staticmethod
    def construct_url_to_info(conversations: List[Tuple[str, List[DialogueTurn]]]) -> Dict[str, StormInformation]:
        url_to_info = {}
        for (persona, conv) in conversations:
            for turn in conv:
                for storm_info in turn.search_results:
                    if storm_info.url in url_to_info:
                        url_to_info[storm_info.url].snippets.extend(storm_info.snippets)
                    else:
                        url_to_info[storm_info.url] = storm_info
        for url in url_to_info:
            url_to_info[url].snippets = list(set(url_to_info[url].snippets))
        logger.info(f"Constructed url_to_info with {len(url_to_info)} unique URLs")
        return url_to_info

    @staticmethod
    def construct_log_dict(conversations: List[Tuple[str, List[DialogueTurn]]]) -> List[Dict[str, Union[str, Any]]]:
        conversation_log = []
        for (persona, conv) in conversations:
            conversation_log.append({
                'perspective': persona,
                'dlg_turns': [turn.log() for turn in conv]
            })
        logger.debug(f"Constructed log dict with {len(conversation_log)} conversations")
        return conversation_log

    def dump_url_to_info(self, path):
        url_to_info = copy.deepcopy(self.url_to_info)
        for url in url_to_info:
            url_to_info[url] = url_to_info[url].to_dict()
        FileIOHelper.dump_json(url_to_info, path)
        logger.info(f"Dumped url_to_info to {path}")

    @classmethod
    def from_conversation_log_file(cls, path):
        conversation_log_data = FileIOHelper.load_json(path)
        conversations = []
        for item in conversation_log_data:
            dialogue_turns = [DialogueTurn(**turn) for turn in item['dlg_turns']]
            persona = item['perspective']
            conversations.append((persona, dialogue_turns))
        logger.info(f"Created StormInformationTable from conversation log file: {path}")
        return cls(conversations)

    def prepare_table_for_retrieval(self):
        logger.info("Preparing table for retrieval")
        if self.embedding_model_name is None:
            raise ValueError("embedding_model_name is not set")
        self.encoder = OllamaEmbeddings(model=self.embedding_model_name)
        self.collected_urls = []
        self.collected_snippets = []
        for url, information in self.url_to_info.items():
            for snippet in information.snippets:
                self.collected_urls.append(url)
                self.collected_snippets.append(snippet)
        if not self.collected_snippets:
            raise ValueError("Collected snippets is empty. Cannot encode.")
        self.encoded_snippets = self.encoder.encode(self.collected_snippets, show_progress_bar=False)
        logger.debug(f"Prepared {len(self.collected_snippets)} snippets for retrieval")

    def retrieve_information(self, queries: Union[List[str], str], search_top_k, embedding_model_name) -> List[StormInformation]:
        logger.info(f"Retrieving information for {len(queries) if isinstance(queries, list) else 1} queries")
        if self.encoder is None:
            self.prepare_table_for_retrieval(embedding_model_name)
        
        selected_urls = []
        selected_snippets = []
        if isinstance(queries, str):
            queries = [queries]
        for query in queries:
            encoded_query = self.encoder.encode(query, show_progress_bar=False)
            sim = cosine_similarity([encoded_query], self.encoded_snippets)[0]
            sorted_indices = np.argsort(sim)
            for i in sorted_indices[-search_top_k:][::-1]:
                selected_urls.append(self.collected_urls[i])
                selected_snippets.append(self.collected_snippets[i])

        url_to_snippets = {}
        for url, snippet in zip(selected_urls, selected_snippets):
            if url not in url_to_snippets:
                url_to_snippets[url] = set()
            url_to_snippets[url].add(snippet)

        selected_url_to_info = {}
        for url in url_to_snippets:
            selected_url_to_info[url] = copy.deepcopy(self.url_to_info[url])
            selected_url_to_info[url].snippets = list(url_to_snippets[url])

        logger.debug(f"Retrieved {len(selected_url_to_info)} pieces of information")
        return list(selected_url_to_info.values())

class StormArticle(Article):
    def __init__(self, topic_name):
        super().__init__(topic_name=topic_name)
        self.reference = {
            "url_to_unified_index": {},
            "url_to_info": {}
        }
        logger.info(f"StormArticle initialized with topic: {topic_name}")

    def find_section(self, node: ArticleSectionNode, name: str) -> Optional[ArticleSectionNode]:
        logger.debug(f"Searching for section: {name}")
        if node.section_name == name:
            return node
        for child in node.children:
            result = self.find_section(child, name)
            if result:
                return result
        return None

    def _merge_new_info_to_references(self, new_info_list: List[StormInformation], index_to_keep=None) -> Dict[int, int]:
        logger.info(f"Merging {len(new_info_list)} new information pieces to references")
        citation_idx_mapping = {}
        for idx, storm_info in enumerate(new_info_list):
            if index_to_keep is not None and idx not in index_to_keep:
                continue
            url = storm_info.url
            if url not in self.reference["url_to_unified_index"]:
                self.reference["url_to_unified_index"][url] = len(self.reference["url_to_unified_index"]) + 1
                self.reference["url_to_info"][url] = storm_info
                logger.debug(f"Added new reference: {url}")
            else:
                existing_snippets = self.reference["url_to_info"][url].snippets
                existing_snippets.extend(storm_info.snippets)
                self.reference["url_to_info"][url].snippets = list(set(existing_snippets))
                logger.debug(f"Updated existing reference: {url}")
            citation_idx_mapping[idx + 1] = self.reference["url_to_unified_index"][url]
        return citation_idx_mapping

    def insert_or_create_section(self, article_dict: Dict[str, Dict], parent_section_name: str = None, trim_children=False):
        logger.info(f"Inserting or creating section under parent: {parent_section_name}")
        parent_node = self.root if parent_section_name is None else self.find_section(self.root, parent_section_name)

        if trim_children:
            section_names = set(article_dict.keys())
            for child in parent_node.children[:]:
                if child.section_name not in section_names:
                    parent_node.remove_child(child)
                    logger.debug(f"Removed child section: {child.section_name}")

        for section_name, content_dict in article_dict.items():
            current_section_node = self.find_section(parent_node, section_name)
            if current_section_node is None:
                current_section_node = ArticleSectionNode(section_name=section_name, content=content_dict["content"].strip())
                insert_to_front = parent_node.section_name == self.root.section_name and current_section_node.section_name == "summary"
                parent_node.add_child(current_section_node, insert_to_front=insert_to_front)
                logger.debug(f"Created new section: {section_name}")
            else:
                current_section_node.content = content_dict["content"].strip()
                logger.debug(f"Updated existing section: {section_name}")

            self.insert_or_create_section(article_dict=content_dict["subsections"], parent_section_name=section_name, trim_children=True)

    def update_section(self, current_section_content: str, current_section_info_list: List[StormInformation], parent_section_name: Optional[str] = None) -> Optional[ArticleSectionNode]:
        logger.info(f"Updating section under parent: {parent_section_name}")
        if current_section_info_list is not None:
            references = set([int(x) for x in re.findall(r'\[(\d+)\]', current_section_content)])
            if len(references) > 0:
                max_ref_num = max(references)
                if max_ref_num > len(current_section_info_list):
                    for i in range(len(current_section_info_list), max_ref_num + 1):
                        current_section_content = current_section_content.replace(f'[{i}]', '')
                        if i in references:
                            references.remove(i)
                    logger.debug(f"Removed {max_ref_num - len(current_section_info_list)} invalid references")
            index_to_keep = [i - 1 for i in references]
            citation_mapping = self._merge_new_info_to_references(current_section_info_list, index_to_keep)
            current_section_content = ArticleTextProcessing.update_citation_index(current_section_content, citation_mapping)
            logger.debug(f"Updated {len(citation_mapping)} citations")

        if parent_section_name is None:
            parent_section_name = self.root.section_name
        article_dict = ArticleTextProcessing.parse_article_into_dict(current_section_content)
        self.insert_or_create_section(article_dict=article_dict, parent_section_name=parent_section_name, trim_children=False)
        logger.info("Section update completed")

    def get_outline_as_list(self, root_section_name: Optional[str] = None, add_hashtags: bool = False, include_root: bool = True) -> List[str]:
        logger.info(f"Getting outline as list from root: {root_section_name}")
        if root_section_name is None:
            section_node = self.root
        else:
            section_node = self.find_section(self.root, root_section_name)
            include_root = include_root or section_node != self.root.section_name
        if section_node is None:
            logger.warning(f"Section not found: {root_section_name}")
            return []
        result = []

        def preorder_traverse(node, level):
            prefix = "#" * level if add_hashtags else ""
            result.append(f"{prefix} {node.section_name}".strip() if add_hashtags else node.section_name)
            for child in node.children:
                preorder_traverse(child, level + 1)

        if include_root:
            preorder_traverse(section_node, level=1)
        else:
            for child in section_node.children:
                preorder_traverse(child, level=1)
        logger.debug(f"Generated outline with {len(result)} sections")
        return result

    def to_string(self) -> str:
        logger.info("Converting article to string")
        result = []

        def preorder_traverse(node, level):
            prefix = "#" * level
            result.append(f"{prefix} {node.section_name}".strip())
            result.append(node.content)
            for child in node.children:
                preorder_traverse(child, level + 1)

        for child in self.root.children:
            preorder_traverse(child, level=1)
        result = [i.strip() for i in result if i is not None and i.strip()]
        logger.debug(f"Generated string representation with {len(result)} lines")
        return "\n\n".join(result)

    def reorder_reference_index(self):
        logger.info("Reordering reference indices")
        ref_indices = []

        def pre_order_find_index(node):
            if node is not None:
                if node.content is not None and node.content:
                    ref_indices.extend(ArticleTextProcessing.parse_citation_indices(node.content))
                for child in node.children:
                    pre_order_find_index(child)

        pre_order_find_index(self.root)
        ref_index_mapping = {}
        for ref_index in ref_indices:
            if ref_index not in ref_index_mapping:
                ref_index_mapping[ref_index] = len(ref_index_mapping) + 1
        
        logger.debug(f"Created reference index mapping for {len(ref_index_mapping)} indices")

        def pre_order_update_index(node):
            if node is not None:
                if node.content is not None and node.content:
                    node.content = ArticleTextProcessing.update_citation_index(node.content, ref_index_mapping)
                for child in node.children:
                    pre_order_update_index(child)

        pre_order_update_index(self.root)
        logger.debug("Updated citation indices in article content")

        for url in list(self.reference["url_to_unified_index"]):
            pre_index = self.reference["url_to_unified_index"][url]
            if pre_index not in ref_index_mapping:
                del self.reference["url_to_unified_index"][url]
                logger.debug(f"Removed unused reference: {url}")
            else:
                new_index = ref_index_mapping[pre_index]
                self.reference["url_to_unified_index"][url] = new_index
                logger.debug(f"Updated reference index for {url}: {pre_index} -> {new_index}")

        logger.info("Reference index reordering completed")

    def get_outline_tree(self):
        logger.info("Generating outline tree")
        def build_tree(node) -> Dict[str, Dict]:
            tree = {}
            for child in node.children:
                tree[child.section_name] = build_tree(child)
            return tree if tree else {}

        outline_tree = build_tree(self.root)
        logger.debug(f"Generated outline tree with {len(outline_tree)} top-level sections")
        return outline_tree

    def get_first_level_section_names(self) -> List[str]:
        logger.info("Getting first level section names")
        section_names = [i.section_name for i in self.root.children]
        logger.debug(f"Found {len(section_names)} first-level sections")
        return section_names

    @classmethod
    def from_outline_file(cls, topic: str, file_path: str):
        logger.info(f"Creating StormArticle from outline file: {file_path}")
        outline_str = FileIOHelper.load_str(file_path)
        return StormArticle.from_outline_str(topic=topic, outline_str=outline_str)

    @classmethod
    def from_outline_str(cls, topic: str, outline_str: str):
        logger.info(f"Creating StormArticle from outline string for topic: {topic}")
        lines = []
        try:
            lines = outline_str.split("\n")
            lines = [line.strip() for line in lines if line.strip()]
        except Exception as e:
            logger.error(f"Error processing outline string: {str(e)}")

        instance = cls(topic)
        if lines:
            adjust_level = lines[0].startswith('#') and lines[0].replace('#', '').strip().lower() == topic.lower().replace("_", " ")
            if adjust_level:
                lines = lines[1:]
                logger.debug("Adjusted outline level")
            node_stack = [(0, instance.root)]

            for line in lines:
                level = line.count('#') - adjust_level
                section_name = line.replace('#', '').strip()

                if section_name == topic:
                    continue

                new_node = ArticleSectionNode(section_name)

                while node_stack and level <= node_stack[-1][0]:
                    node_stack.pop()

                node_stack[-1][1].add_child(new_node)
                node_stack.append((level, new_node))
                logger.debug(f"Added section: {section_name} at level {level}")

        logger.info(f"StormArticle created from outline with {len(instance.get_first_level_section_names())} top-level sections")
        return instance

    def dump_outline_to_file(self, file_path):
        logger.info(f"Dumping outline to file: {file_path}")
        outline = self.get_outline_as_list(add_hashtags=True, include_root=False)
        FileIOHelper.write_str("\n".join(outline), file_path)
        logger.debug(f"Outline with {len(outline)} sections written to file")

    def dump_reference_to_file(self, file_path):
        logger.info(f"Dumping references to file: {file_path}")
        reference = copy.deepcopy(self.reference)
        for url in reference["url_to_info"]:
            reference["url_to_info"][url] = reference["url_to_info"][url].to_dict()
        FileIOHelper.dump_json(reference, file_path)
        logger.debug(f"Dumped {len(reference['url_to_info'])} references to file")

    def dump_article_as_plain_text(self, file_path):
        logger.info(f"Dumping article as plain text to file: {file_path}")
        text = self.to_string()
        FileIOHelper.write_str(text, file_path)
        logger.debug(f"Article text of length {len(text)} written to file")

    @classmethod
    def from_string(cls, topic_name: str, article_text: str, references: dict):
        logger.info(f"Creating StormArticle from string for topic: {topic_name}")
        article_dict = ArticleTextProcessing.parse_article_into_dict(article_text)
        article = cls(topic_name=topic_name)
        article.insert_or_create_section(article_dict=article_dict)
        for url in list(references["url_to_info"]):
            references["url_to_info"][url] = StormInformation.from_dict(references["url_to_info"][url])
        article.reference = references
        logger.debug(f"Created StormArticle with {len(article.get_first_level_section_names())} top-level sections and {len(references['url_to_info'])} references")
        return article

    def post_processing(self):
        logger.info("Starting post-processing of StormArticle")
        self.prune_empty_nodes()
        logger.debug("Empty nodes pruned")
        self.reorder_reference_index()
        logger.debug("Reference indices reordered")
        logger.info("Post-processing completed")

# Voeg een handler toe om logs naar een bestand te schrijven
file_handler = logging.FileHandler('storm_article.log')
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
logger.addHandler(file_handler)

logger.info("StormArticle module initialized and logging configured")