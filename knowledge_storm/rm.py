import logging
import os
from typing import Callable, Union, List, Dict, Tuple

import dspy
import pandas as pd
import requests

from langchain.document_loaders import (
    CSVLoader,
    UnstructuredExcelLoader,
    UnstructuredWordDocumentLoader,
    UnstructuredPDFLoader,
    TextLoader
)
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
import streamlit as st

OLLAMA_MODEL_NAME = st.secrets["OLLAMA_MODEL_NAME"]
EMBEDDING_MODEL_NAME = st.secrets["EMBEDDING_MODEL_NAME"]



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

logger = logging.getLogger(__name__)

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
        self.embeddings = embeddings or OllamaEmbeddings(model=EMBEDDING_MODEL_NAME)
        logger.info(f"VectorRM initialized with OllamaEmbeddings model: {self.embeddings.model}")
        
        try:
            self.vectorstore = Chroma(
                persist_directory=self.persist_directory,
                embedding_function=self.embeddings,
                collection_name=self.collection_name
            )
            logger.info(f"Chroma vectorstore successfully initialized")
        except Exception as e:
            logger.error(f"Error initializing Chroma vectorstore: {e}")
            self.vectorstore = None

    def add_directory(self, directory_path: str):
        """
        Adds all supported files from a directory to the vectorstore.
        """
        supported_extensions = ['.csv', '.xlsx', '.docx', '.pdf', '.txt']
        
        for root, _, files in os.walk(directory_path):
            for file in files:
                file_path = os.path.join(root, file)
                _, ext = os.path.splitext(file)
                if ext.lower() in supported_extensions:
                    self.add_file(file_path)

    def add_file(self, file_path: str):
        """
        Adds a single file to the vectorstore based on its file type.
        """
        _, ext = os.path.splitext(file_path)
        ext = ext.lower()

        try:
            if ext == '.csv':
                loader = CSVLoader(file_path)
            elif ext == '.xlsx':
                loader = UnstructuredExcelLoader(file_path)
            elif ext == '.docx':
                loader = UnstructuredWordDocumentLoader(file_path)
            elif ext == '.pdf':
                loader = UnstructuredPDFLoader(file_path)
            elif ext == '.txt':
                loader = TextLoader(file_path)
            else:
                logger.warning(f"Unsupported file type: {ext}")
                return

            documents = loader.load()
            self.add_documents(documents)
            logger.info(f"Successfully added documents from file: {file_path}")
        except Exception as e:
            logger.error(f"Error adding file {file_path}: {e}")

    def add_documents(self, documents: List[Document]):
        if self.vectorstore is not None:
            self.vectorstore.add_documents(documents)
            logger.info(f"Successfully added {len(documents)} documents to the vectorstore")
        else:
            logger.error("Vectorstore is not initialized. Cannot add documents.")

    def get_usage_and_reset(self):
        usage = self.usage
        self.usage = 0
        return {'VectorRM': usage}

    def retrieve(self, query_or_queries: Union[str, List[str]]) -> List[Dict]:
        queries = [query_or_queries] if isinstance(query_or_queries, str) else query_or_queries
        self.usage += len(queries)
        
        collected_results = []
        for query in queries:
            try:
                results = self.vectorstore.similarity_search_with_score(query, k=self.k)
                
                for doc, score in results:
                    collected_results.append({
                        'query': query,
                        'page_content': doc.page_content,
                        'metadata': doc.metadata,
                        'score': score
                    })
                
                logger.info(f"VectorRM retrieved results for query '{query}'")
            except Exception as e:
                logger.error(f'VectorRM: Error occurred when retrieving for query {query}: {e}')
        return collected_results

class VectorRMRetriever:
    def __init__(self, embeddings, k: int = 3):
        self.vector_rm = VectorRM(embeddings=embeddings, k=k)
        self.k = k
        logger.info(f"VectorRMRetriever initialized with k: {k}")

    def retrieve(self, query: Union[str, List[str]]) -> List[Dict]:
        logger.info(f"VectorRMRetriever.retrieve called with query: {query}")
        
        results = self.vector_rm.retrieve(query)
        logger.info(f"Number of retrieved VectorRM results: {len(results)}")
        return results[:self.k]

    def add_directory(self, directory_path: str):
        """
        Adds all supported files from a directory to the VectorRM.
        """
        self.vector_rm.add_directory(directory_path)

    def add_file(self, file_path: str):
        """
        Adds a single file to the VectorRM.
        """
        self.vector_rm.add_file(file_path)