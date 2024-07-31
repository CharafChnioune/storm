import logging
import os
from typing import Callable, Union, List, Dict, Tuple

import dspy
import pandas as pd
import requests

from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from .utils import WebPageHelper


class YouRM(dspy.Retrieve):
    def __init__(self, ydc_api_key=None, k=3, is_valid_source: Callable = None):
        super().__init__(k=k)
        if not ydc_api_key and not os.environ.get("YDC_API_KEY"):
            raise RuntimeError("You must supply ydc_api_key or set environment variable YDC_API_KEY")
        elif ydc_api_key:
            self.ydc_api_key = ydc_api_key
        else:
            self.ydc_api_key = os.environ["YDC_API_KEY"]
        self.usage = 0

        # If not None, is_valid_source shall be a function that takes a URL and returns a boolean.
        if is_valid_source:
            self.is_valid_source = is_valid_source
        else:
            self.is_valid_source = lambda x: True

    def get_usage_and_reset(self):
        usage = self.usage
        self.usage = 0

        return {'YouRM': usage}

    def forward(self, query_or_queries: Union[str, List[str]], exclude_urls: List[str] = []):
        """Search with You.com for self.k top passages for query or queries

        Args:
            query_or_queries (Union[str, List[str]]): The query or queries to search for.
            exclude_urls (List[str]): A list of urls to exclude from the search results.

        Returns:
            a list of Dicts, each dict has keys of 'description', 'snippets' (list of strings), 'title', 'url'
        """
        queries = (
            [query_or_queries]
            if isinstance(query_or_queries, str)
            else query_or_queries
        )
        self.usage += len(queries)
        collected_results = []
        for query in queries:
            try:
                headers = {"X-API-Key": self.ydc_api_key}
                results = requests.get(
                    f"https://api.ydc-index.io/search?query={query}",
                    headers=headers,
                ).json()

                authoritative_results = []
                for r in results['hits']:
                    if self.is_valid_source(r['url']) and r['url'] not in exclude_urls:
                        authoritative_results.append(r)
                if 'hits' in results:
                    collected_results.extend(authoritative_results[:self.k])
            except Exception as e:
                logging.error(f'Error occurs when searching query {query}: {e}')

        return collected_results


class BingSearch(dspy.Retrieve):
    def __init__(self, bing_search_api_key=None, k=3, is_valid_source: Callable = None,
                 min_char_count: int = 150, snippet_chunk_size: int = 1000, webpage_helper_max_threads=10,
                 mkt='en-US', language='en', **kwargs):
        """
        Params:
            min_char_count: Minimum character count for the article to be considered valid.
            snippet_chunk_size: Maximum character count for each snippet.
            webpage_helper_max_threads: Maximum number of threads to use for webpage helper.
            mkt, language, **kwargs: Bing search API parameters.
            - Reference: https://learn.microsoft.com/en-us/bing/search-apis/bing-web-search/reference/query-parameters
        """
        super().__init__(k=k)
        if not bing_search_api_key and not os.environ.get("BING_SEARCH_API_KEY"):
            raise RuntimeError(
                "You must supply bing_search_subscription_key or set environment variable BING_SEARCH_API_KEY")
        elif bing_search_api_key:
            self.bing_api_key = bing_search_api_key
        else:
            self.bing_api_key = os.environ["BING_SEARCH_API_KEY"]
        self.endpoint = "https://api.bing.microsoft.com/v7.0/search"
        self.params = {
            'mkt': mkt,
            "setLang": language,
            "count": k,
            **kwargs
        }
        self.webpage_helper = WebPageHelper(
            min_char_count=min_char_count,
            snippet_chunk_size=snippet_chunk_size,
            max_thread_num=webpage_helper_max_threads
        )
        self.usage = 0

        # If not None, is_valid_source shall be a function that takes a URL and returns a boolean.
        if is_valid_source:
            self.is_valid_source = is_valid_source
        else:
            self.is_valid_source = lambda x: True

    def get_usage_and_reset(self):
        usage = self.usage
        self.usage = 0

        return {'BingSearch': usage}

    def forward(self, query_or_queries: Union[str, List[str]], exclude_urls: List[str] = []):
        """Search with Bing for self.k top passages for query or queries

        Args:
            query_or_queries (Union[str, List[str]]): The query or queries to search for.
            exclude_urls (List[str]): A list of urls to exclude from the search results.

        Returns:
            a list of Dicts, each dict has keys of 'description', 'snippets' (list of strings), 'title', 'url'
        """
        queries = (
            [query_or_queries]
            if isinstance(query_or_queries, str)
            else query_or_queries
        )
        self.usage += len(queries)

        url_to_results = {}

        headers = {"Ocp-Apim-Subscription-Key": self.bing_api_key}

        for query in queries:
            try:
                results = requests.get(
                    self.endpoint,
                    headers=headers,
                    params={**self.params, 'q': query}
                ).json()

                for d in results['webPages']['value']:
                    if self.is_valid_source(d['url']) and d['url'] not in exclude_urls:
                        url_to_results[d['url']] = {'url': d['url'], 'title': d['name'], 'description': d['snippet']}
            except Exception as e:
                logging.error(f'Error occurs when searching query {query}: {e}')

        valid_url_to_snippets = self.webpage_helper.urls_to_snippets(list(url_to_results.keys()))
        collected_results = []
        for url in valid_url_to_snippets:
            r = url_to_results[url]
            r['snippets'] = valid_url_to_snippets[url]['snippets']
            collected_results.append(r)

        return collected_results

class VectorRM:
    def __init__(
        self,
        collection_name: str = "my_documents",
        embeddings: OllamaEmbeddings = None,
        k: int = 3,
        persist_directory: str = "./chroma_db"
    ):
        self.collection_name = collection_name
        self.k = k
        self.persist_directory = persist_directory
        self.usage = 0
        self.embeddings = embeddings or OllamaEmbeddings(model="llama2")
        logging.info(f"VectorRM geïnitialiseerd met OllamaEmbeddings model: {self.embeddings.model}")
        
        try:
            self.vectorstore = Chroma(
                persist_directory=self.persist_directory,
                embedding_function=self.embeddings,
                collection_name=self.collection_name
            )
            logging.info(f"Chroma vectorstore succesvol geïnitialiseerd")
        except Exception as e:
            logging.error(f"Fout bij initialiseren van Chroma vectorstore: {e}")
            self.vectorstore = None

    def add_csv(self, file_path: str):
        try:
            # Gebruik het correcte pad naar het CSV-bestand
            csv_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", file_path)
            if not os.path.exists(csv_path):
                raise FileNotFoundError(f"CSV-bestand niet gevonden: {csv_path}")
            
            df = pd.read_csv(csv_path)
            documents = []
            for _, row in df.iterrows():
                content = " ".join(f"{col}: {val}" for col, val in row.items() if pd.notna(val))
                documents.append(Document(page_content=content))
            
            self.add_documents(documents)
            logging.info(f"Succesvol {len(documents)} documenten toegevoegd uit CSV-bestand: {csv_path}")
        except Exception as e:
            logging.error(f"Fout bij het toevoegen van CSV-bestand {csv_path}: {e}")

    # De rest van de methoden blijven ongewijzigd
    def add_documents(self, documents):
        if self.vectorstore is not None:
            self.vectorstore.add_documents(documents)
            logging.info(f"Succesvol {len(documents)} documenten toegevoegd aan de vectorstore")
        else:
            logging.error("Vectorstore is niet geïnitialiseerd. Kan geen documenten toevoegen.")

    def get_usage_and_reset(self):
        usage = self.usage
        self.usage = 0
        return {'VectorRM': usage}

    def forward(self, query_or_queries: Union[str, List[str]], exclude_urls: List[str] = []):
        queries = [query_or_queries] if isinstance(query_or_queries, str) else query_or_queries
        self.usage += len(queries)
        
        collected_results = []
        for query in queries:
            try:
                results = self.vectorstore.similarity_search_with_score(query, k=self.k)
                
                answer = self.process_results(query, results)
                
                collected_results.append({
                    'query': query,
                    'answer': answer,
                    'page_content': answer,  # Voor compatibiliteit met de bestaande structuur
                    'metadata': {}  # Lege metadata voor compatibiliteit
                })
                
                logging.info(f"VectorRM antwoord gegenereerd voor query '{query}'")
            except Exception as e:
                logging.error(f'VectorRM: Fout opgetreden bij het beantwoorden van query {query}: {e}')
        return collected_results

    def process_results(self, query: str, results):
        if "hoevaak" in query.lower() and "genoemd" in query.lower():
            word_to_count = query.lower().split("hoevaak")[1].split("genoemd")[0].strip()
            count = sum(doc.page_content.lower().count(word_to_count) for doc, _ in results)
            return f"Het woord '{word_to_count}' komt {count} keer voor in de relevante documenten."
        else:
            relevant_content = " ".join([doc.page_content for doc, _ in results])
            return f"Gebaseerd op de relevante documenten: {relevant_content[:200]}..."