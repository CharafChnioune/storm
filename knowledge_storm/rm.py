import logging
import os
from typing import Callable, Union, List

import backoff
import dspy
import requests
from dsp import backoff_hdlr, giveup_hdlr

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import Qdrant
from qdrant_client import QdrantClient

from .utils import WebPageHelper


class YouRM(dspy.Retrieve):
    def __init__(self, ydc_api_key=None, k=3, is_valid_source: Callable = None):
        super().__init__(k=k)
        # Controleer of er een API-sleutel is opgegeven of in de omgevingsvariabelen staat
        if not ydc_api_key and not os.environ.get("YDC_API_KEY"):
            raise RuntimeError(
                "Je moet een ydc_api_key opgeven of de omgevingsvariabele YDC_API_KEY instellen"
            )
        elif ydc_api_key:
            self.ydc_api_key = ydc_api_key
        else:
            self.ydc_api_key = os.environ["YDC_API_KEY"]
        self.usage = 0

        # Functie om te controleren of een bron geldig is
        if is_valid_source:
            self.is_valid_source = is_valid_source
        else:
            self.is_valid_source = lambda x: True

    # Methode om het gebruik te resetten en te rapporteren
    def get_usage_and_reset(self):
        """
        Haalt het huidige gebruik op en reset het naar nul.

        Returns:
            dict: Een dictionary met de sleutel "YouRM" en het huidige gebruik als waarde.
        """
        usage = self.usage
        self.usage = 0

        return {"YouRM": usage}

    # Hoofdmethode voor het uitvoeren van zoekopdrachten
    def forward(
        self, query_or_queries: Union[str, List[str]], exclude_urls: List[str] = []
    ):
        """Zoek met You.com naar de top self.k passages voor query of queries

        Args:
            query_or_queries (Union[str, List[str]]): De query of queries om naar te zoeken.
            exclude_urls (List[str]): Een lijst met urls om uit te sluiten van de zoekresultaten.

        Returns:
            een lijst van Dicts, elke dict heeft sleutels 'description', 'snippets' (lijst van strings), 'title', 'url'
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
                for r in results["hits"]:
                    if self.is_valid_source(r["url"]) and r["url"] not in exclude_urls:
                        authoritative_results.append(r)
                if "hits" in results:
                    collected_results.extend(authoritative_results[: self.k])
            except Exception as e:
                logging.error(f"Fout bij het zoeken naar query {query}: {e}")

        return collected_results


class BingSearch(dspy.Retrieve):
    def __init__(
        self,
        bing_search_api_key=None,
        k=3,
        is_valid_source: Callable = None,
        min_char_count: int = 150,
        snippet_chunk_size: int = 1000,
        webpage_helper_max_threads=10,
        mkt="en-US",
        language="en",
        **kwargs,
    ):
        """
        Parameters:
            min_char_count: Minimum aantal tekens voor een artikel om als geldig te worden beschouwd.
            snippet_chunk_size: Maximaal aantal tekens voor elk snippet.
            webpage_helper_max_threads: Maximaal aantal threads voor de webpage helper.
            mkt, language, **kwargs: Bing search API parameters.
            - Referentie: https://learn.microsoft.com/en-us/bing/search-apis/bing-web-search/reference/query-parameters
        """
        super().__init__(k=k)
        # Controleer of er een API-sleutel is opgegeven of in de omgevingsvariabelen staat
        if not bing_search_api_key and not os.environ.get("BING_SEARCH_API_KEY"):
            raise RuntimeError(
                "Je moet bing_search_subscription_key opgeven of de omgevingsvariabele BING_SEARCH_API_KEY instellen"
            )
        elif bing_search_api_key:
            self.bing_api_key = bing_search_api_key
        else:
            self.bing_api_key = os.environ["BING_SEARCH_API_KEY"]
        self.endpoint = "https://api.bing.microsoft.com/v7.0/search"
        self.params = {"mkt": mkt, "setLang": language, "count": k, **kwargs}
        self.webpage_helper = WebPageHelper(
            min_char_count=min_char_count,
            snippet_chunk_size=snippet_chunk_size,
            max_thread_num=webpage_helper_max_threads,
        )
        self.usage = 0

        # Functie om te controleren of een bron geldig is
        if is_valid_source:
            self.is_valid_source = is_valid_source
        else:
            self.is_valid_source = lambda x: True

    # Methode om het gebruik te resetten en te rapporteren
    def get_usage_and_reset(self):
        usage = self.usage
        self.usage = 0

        return {"BingSearch": usage}

    # Hoofdmethode voor het uitvoeren van zoekopdrachten
    def forward(
        self, query_or_queries: Union[str, List[str]], exclude_urls: List[str] = []
    ):
        """Zoek met Bing naar de top self.k passages voor query of queries

        Args:
            query_or_queries (Union[str, List[str]]): De query of queries om naar te zoeken.
            exclude_urls (List[str]): Een lijst met urls om uit te sluiten van de zoekresultaten.

        Returns:
            een lijst van Dicts, elke dict heeft sleutels 'description', 'snippets' (lijst van strings), 'title', 'url'
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
                    self.endpoint, headers=headers, params={**self.params, "q": query}
                ).json()

                for d in results["webPages"]["value"]:
                    if self.is_valid_source(d["url"]) and d["url"] not in exclude_urls:
                        url_to_results[d["url"]] = {
                            "url": d["url"],
                            "title": d["name"],
                            "description": d["snippet"],
                        }
            except Exception as e:
                logging.error(f"Fout bij het zoeken naar query {query}: {e}")

        valid_url_to_snippets = self.webpage_helper.urls_to_snippets(
            list(url_to_results.keys())
        )
        collected_results = []
        for url in valid_url_to_snippets:
            r = url_to_results[url]
            r["snippets"] = valid_url_to_snippets[url]["snippets"]
            collected_results.append(r)

        return collected_results


class VectorRM(dspy.Retrieve):
    """Haal informatie op uit aangepaste documenten met behulp van Qdrant.

    Om compatibel te zijn met STORM, moeten de aangepaste documenten de volgende velden hebben:
        - content: De hoofdtekstinhoud van het document.
        - title: De titel van het document.
        - url: De URL van het document. STORM gebruikt url als de unieke identifier van het document, dus zorg ervoor dat verschillende
            documenten verschillende url's hebben.
        - description (optioneel): De beschrijving van het document.
    De documenten moeten worden opgeslagen in een CSV-bestand.
    """

    def __init__(
        self,
        collection_name: str,
        embedding_model: str,
        device: str = "mps",
        k: int = 3,
    ):
        """
        Parameters:
            collection_name: Naam van de Qdrant-collectie.
            embedding_model: Naam van het Hugging Face embedding model.
            device: Apparaat om het embedding model op uit te voeren, kan "mps", "cuda", "cpu" zijn.
            k: Aantal top chunks om op te halen.
        """
        super().__init__(k=k)
        self.usage = 0
        # Controleer of de collectie is opgegeven
        if not collection_name:
            raise ValueError("Geef een collectienaam op.")
        # Controleer of het embedding model is opgegeven
        if not embedding_model:
            raise ValueError("Geef een embedding model op.")

        model_kwargs = {"device": device}
        encode_kwargs = {"normalize_embeddings": True}
        self.model = HuggingFaceEmbeddings(
            model_name=embedding_model,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs,
        )

        self.collection_name = collection_name
        self.client = None
        self.qdrant = None

    def _check_collection(self):
        """
        Controleer of de Qdrant-collectie bestaat en maak deze aan als dat niet het geval is.
        """
        if self.client is None:
            raise ValueError("Qdrant-client is niet geïnitialiseerd.")
        if self.client.collection_exists(collection_name=f"{self.collection_name}"):
            print(
                f"Collectie {self.collection_name} bestaat. De collectie wordt geladen..."
            )
            self.qdrant = Qdrant(
                client=self.client,
                collection_name=self.collection_name,
                embeddings=self.model,
            )
        else:
            raise ValueError(
                f"Collectie {self.collection_name} bestaat niet. Maak eerst de collectie aan."
            )

    def init_online_vector_db(self, url: str, api_key: str):
        """
        Initialiseer de Qdrant-client die verbonden is met een online vector store met de gegeven URL en API-sleutel.

        Args:
            url (str): URL van de Qdrant-server.
            api_key (str): API-sleutel voor de Qdrant-server.
        """
        if api_key is None:
            if not os.getenv("QDRANT_API_KEY"):
                raise ValueError("Geef een api-sleutel op.")
            api_key = os.getenv("QDRANT_API_KEY")
        if url is None:
            raise ValueError("Geef een url op voor de Qdrant-server.")

        try:
            self.client = QdrantClient(url=url, api_key=api_key)
            self._check_collection()
        except Exception as e:
            raise ValueError(f"Fout bij het verbinden met de server: {e}")

    def init_offline_vector_db(self, vector_store_path: str):
        """
        Initialiseer de Qdrant-client die verbonden is met een offline vector store met het gegeven vector store mappad.

        Args:
            vector_store_path (str): Pad naar de vector store.
        """
        if vector_store_path is None:
            raise ValueError("Geef een mappad op.")

        try:
            self.client = QdrantClient(path=vector_store_path)
            self._check_collection()
        except Exception as e:
            raise ValueError(f"Fout bij het laden van de vector store: {e}")

    def get_usage_and_reset(self):
        usage = self.usage
        self.usage = 0

        return {"VectorRM": usage}

    def get_vector_count(self):
        """
        Haal het aantal vectoren in de collectie op.

        Returns:
            int: Aantal vectoren in de collectie.
        """
        return self.qdrant.client.count(collection_name=self.collection_name)

    def forward(self, query_or_queries: Union[str, List[str]], exclude_urls: List[str]):
        """
        Zoek in je gegevens naar de top self.k passages voor query of queries.

        Args:
            query_or_queries (Union[str, List[str]]): De query of queries om naar te zoeken.
            exclude_urls (List[str]): Dummy parameter om de interface te matchen. Heeft geen effect.

        Returns:
            een lijst van Dicts, elke dict heeft sleutels 'description', 'snippets' (lijst van strings), 'title', 'url'
        """
        queries = (
            [query_or_queries]
            if isinstance(query_or_queries, str)
            else query_or_queries
        )
        self.usage += len(queries)
        collected_results = []
        for query in queries:
            related_docs = self.qdrant.similarity_search_with_score(query, k=self.k)
            for i in range(len(related_docs)):
                doc = related_docs[i][0]
                collected_results.append(
                    {
                        "description": doc.metadata["description"],
                        "snippets": [doc.page_content],
                        "title": doc.metadata["title"],
                        "url": doc.metadata["url"],
                    }
                )

        return collected_results


class StanfordOvalArxivRM(dspy.Retrieve):
    """[Alpha] Deze retrieval klasse is alleen voor intern gebruik, niet bedoeld voor het publiek."""

    def __init__(self, endpoint, k=3):
        super().__init__(k=k)
        self.endpoint = endpoint
        self.usage = 0

    def get_usage_and_reset(self):
        """
        Reset het gebruik en rapporteer het.

        Returns:
            dict: Een dictionary met het gebruik van CS224vArxivRM.
        """
        usage = self.usage
        self.usage = 0

        return {"CS224vArxivRM": usage}

    def _retrieve(self, query: str):
        """
        Haalt informatie op uit de Stanford Oval Arxiv database voor de gegeven query.

        Args:
            query (str): De zoekopdracht om informatie voor op te halen.

        Returns:
            list: Een lijst van resultaten, elk als een dictionary met relevante informatie.

        Raises:
            Exception: Als er een fout optreedt bij het ophalen van de resultaten.
        """
        payload = {"query": query, "num_blocks": self.k}

        response = requests.post(
            self.endpoint, json=payload, headers={"Content-Type": "application/json"}
        )

        # Controleer of het verzoek succesvol was
        if response.status_code == 200:
            data = response.json()[0]
            results = []
            for i in range(len(data["title"])):
                result = {
                    "title": data["title"][i],
                    "url": data["title"][i],
                    "snippets": [data["text"][i]],
                    "description": "N/A",
                    "meta": {"section_title": data["full_section_title"][i]},
                }
                results.append(result)

            return results
        else:
            raise Exception(
                f"Fout: Kan geen resultaten ophalen. Statuscode: {response.status_code}"
            )

    def forward(
        self, query_or_queries: Union[str, List[str]], exclude_urls: List[str] = []
    ):
        """
        Voert zoekopdrachten uit en verzamelt de resultaten.

        Args:
            query_or_queries (Union[str, List[str]]): De query of queries om naar te zoeken.
            exclude_urls (List[str], optional): Een lijst met urls om uit te sluiten van de zoekresultaten. Standaard is een lege lijst.

        Returns:
            list: Een lijst van verzamelde resultaten.
        """
        collected_results = []
        queries = (
            [query_or_queries]
            if isinstance(query_or_queries, str)
            else query_or_queries
        )

        for query in queries:
            try:
                results = self._retrieve(query)
                collected_results.extend(results)
            except Exception as e:
                logging.error(f"Fout bij het zoeken naar query {query}: {e}")
        return collected_results


class SerperRM(dspy.Retrieve):
    """Haal informatie op uit aangepaste queries met behulp van Serper.dev."""

    def __init__(
        self,
        serper_search_api_key=None,
        k=3,
        query_params=None,
        ENABLE_EXTRA_SNIPPET_EXTRACTION=False,
        min_char_count: int = 150,
        snippet_chunk_size: int = 1000,
        webpage_helper_max_threads=10,
    ):
        """Args:
        serper_search_api_key str: API-sleutel om serper uit te voeren, kan worden gevonden door een account aan te maken op https://serper.dev/
        query_params (dict of lijst van dict): parameters in dictionary of lijst van dictionaries met een maximale grootte van 100 die zullen worden gebruikt voor de query.
            Veelgebruikte velden zijn als volgt (zie meer informatie in https://serper.dev/playground):
                q str: query die zal worden gebruikt met google search
                type str: type dat zal worden gebruikt voor het browsen van google. Types zijn search, images, video, maps, places, etc.
                gl str: Land dat de focus zal zijn voor de zoekopdracht
                location str: Land waar de zoekopdracht vandaan zal komen. Alle locaties kunnen hier worden gevonden: https://api.serper.dev/locations.
                autocorrect bool: Schakel autocorrectie in voor de queries tijdens het zoeken, als de query verkeerd gespeld is, wordt deze bijgewerkt.
                results int: Maximaal aantal resultaten per pagina.
                page int: Maximaal aantal pagina's per aanroep.
                tbs str: datum tijdbereik, standaard ingesteld op elk moment.
                qdr:h str: Datum tijdbereik voor het afgelopen uur.
                qdr:d str: Datum tijdbereik voor de afgelopen 24 uur.
                qdr:w str: Datum tijdbereik voor de afgelopen week.
                qdr:m str: Datum tijdbereik voor de afgelopen maand.
                qdr:y str: Datum tijdbereik voor het afgelopen jaar.
        """
        super().__init__(k=k)
        self.usage = 0
        self.query_params = None
        self.ENABLE_EXTRA_SNIPPET_EXTRACTION = ENABLE_EXTRA_SNIPPET_EXTRACTION
        self.webpage_helper = WebPageHelper(
            min_char_count=min_char_count,
            snippet_chunk_size=snippet_chunk_size,
            max_thread_num=webpage_helper_max_threads,
        )

        if query_params is None:
            self.query_params = {"num": k, "autocorrect": True, "page": 1}
        else:
            self.query_params = query_params
            self.query_params.update({"num": k})
        self.serper_search_api_key = serper_search_api_key
        if not self.serper_search_api_key and not os.environ.get("SERPER_API_KEY"):
            raise RuntimeError(
                "Je moet een serper_search_api_key parameter opgeven of de omgevingsvariabele SERPER_API_KEY instellen"
            )

        elif self.serper_search_api_key:
            self.serper_search_api_key = serper_search_api_key

        else:
            self.serper_search_api_key = os.environ["SERPER_API_KEY"]

        self.base_url = "https://google.serper.dev"

    # Methode om de Serper API aan te roepen
    def serper_runner(self, query_params):
        self.search_url = f"{self.base_url}/search"

        headers = {
            "X-API-KEY": self.serper_search_api_key,
            "Content-Type": "application/json",
        }

        response = requests.request(
            "POST", self.search_url, headers=headers, json=query_params
        )

        if response == None:
            raise RuntimeError(
                f"Er is een fout opgetreden tijdens het uitvoeren van het zoekproces.\n De fout is {response.reason}, mislukt met statuscode {response.status_code}"
            )

        return response.json()

    # Methode om het gebruik te resetten en te rapporteren
    def get_usage_and_reset(self):
        usage = self.usage
        self.usage = 0
        return {"SerperRM": usage}

    # Hoofdmethode voor het uitvoeren van zoekopdrachten
    def forward(self, query_or_queries: Union[str, List[str]], exclude_urls: List[str]):
        """
        Roept de API aan en zoekt naar de doorgegeven query.


        Args:
            query_or_queries (Union[str, List[str]]): De query of queries om naar te zoeken.
            exclude_urls (List[str]): Dummy parameter om de interface te matchen. Heeft geen effect.

        Returns:
            een lijst van dictionaries, elke dictionary heeft sleutels 'description', 'snippets' (lijst van strings), 'title', 'url'
        """
        queries = (
            [query_or_queries]
            if isinstance(query_or_queries, str)
            else query_or_queries
        )

        self.usage += len(queries)
        self.results = []
        collected_results = []
        for query in queries:
            if query == "Queries:":
                continue
            query_params = self.query_params

            # Alle beschikbare parameters kunnen worden gevonden in de playground: https://serper.dev/playground
            # Stelt de json-waarde voor query in op de query die wordt geparseerd.
            query_params["q"] = query

            # Stelt het type in op search, kan images, video, places, maps etc zijn die Google biedt.
            query_params["type"] = "search"

            self.result = self.serper_runner(query_params)
            self.results.append(self.result)

        # Array van dictionaries die door Storm zullen worden gebruikt om de jsons te maken
        collected_results = []

        if self.ENABLE_EXTRA_SNIPPET_EXTRACTION:
            urls = []
            for result in self.results:
                organic_results = result.get("organic", [])
                for organic in organic_results:
                    url = organic.get("link")
                    if url:
                        urls.append(url)
            valid_url_to_snippets = self.webpage_helper.urls_to_snippets(urls)
        else:
            valid_url_to_snippets = {}

        for result in self.results:
            try:
                # Een array van dictionaries die de snippets, titel van het document en url bevat die zullen worden gebruikt.
                organic_results = result.get("organic")
                knowledge_graph = result.get("knowledgeGraph")
                for organic in organic_results:
                    snippets = [organic.get("snippet")]
                    if self.ENABLE_EXTRA_SNIPPET_EXTRACTION:
                        snippets.extend(
                            valid_url_to_snippets.get(url, {}).get("snippets", [])
                        )
                    collected_results.append(
                        {
                            "snippets": snippets,
                            "title": organic.get("title"),
                            "url": organic.get("link"),
                            "description": (
                                knowledge_graph.get("description")
                                if knowledge_graph is not None
                                else ""
                            ),
                        }
                    )
            except:
                continue

        return collected_results


class BraveRM(dspy.Retrieve):
    def __init__(
        self, brave_search_api_key=None, k=3, is_valid_source: Callable = None
    ):
        super().__init__(k=k)
        if not brave_search_api_key and not os.environ.get("BRAVE_API_KEY"):
            raise RuntimeError(
                "Je moet brave_search_api_key opgeven of de omgevingsvariabele BRAVE_API_KEY instellen"
            )
        elif brave_search_api_key:
            self.brave_search_api_key = brave_search_api_key
        else:
            self.brave_search_api_key = os.environ["BRAVE_API_KEY"]
        self.usage = 0

        # Functie om te controleren of een bron geldig is
        if is_valid_source:
            self.is_valid_source = is_valid_source
        else:
            self.is_valid_source = lambda x: True

    # Methode om het gebruik te resetten en te rapporteren
    def get_usage_and_reset(self):
        """
        Haalt het huidige gebruik op en reset het naar nul.

        Returns:
            dict: Een dictionary met de sleutel "BraveRM" en het huidige gebruik als waarde.
        """
        usage = self.usage
        self.usage = 0

        return {"BraveRM": usage}

    # Hoofdmethode voor het uitvoeren van zoekopdrachten
    def forward(
        self, query_or_queries: Union[str, List[str]], exclude_urls: List[str] = []
    ):
        """Zoek met api.search.brave.com naar de top self.k passages voor query of queries

        Args:
            query_or_queries (Union[str, List[str]]): De query of queries om naar te zoeken.
            exclude_urls (List[str]): Een lijst met urls om uit te sluiten van de zoekresultaten.

        Returns:
            een lijst van Dicts, elke dict heeft sleutels 'description', 'snippets' (lijst van strings), 'title', 'url'
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
                headers = {
                    "Accept": "application/json",
                    "Accept-Encoding": "gzip",
                    "X-Subscription-Token": self.brave_search_api_key,
                }
                response = requests.get(
                    f"https://api.search.brave.com/res/v1/web/search?result_filter=web&q={query}",
                    headers=headers,
                ).json()
                results = response.get("web", {}).get("results", [])

                for result in results:
                    collected_results.append(
                        {
                            "snippets": result.get("extra_snippets", []),
                            "title": result.get("title"),
                            "url": result.get("url"),
                            "description": result.get("description"),
                        }
                    )
            except Exception as e:
                logging.error(f"Fout bij het zoeken naar query {query}: {e}")

        return collected_results


class SearXNG(dspy.Retrieve):
    def __init__(
        self,
        searxng_api_url,
        searxng_api_key=None,
        k=3,
        is_valid_source: Callable = None,
    ):
        """Initialiseer de SearXNG zoek retriever.
        Stel SearXNG in volgens https://docs.searxng.org/index.html.

        Args:
            searxng_api_url (str): De URL van de SearXNG API. Raadpleeg de SearXNG-documentatie voor details.
            searxng_api_key (str, optioneel): De API-sleutel voor de SearXNG API. Standaard is None. Raadpleeg de SearXNG-documentatie voor details.
            k (int, optioneel): Het aantal top passages om op te halen. Standaard is 3.
            is_valid_source (Callable, optioneel): Een functie die een URL accepteert en een boolean teruggeeft die aangeeft of de
            bron geldig is. Standaard is None.
        """
        super().__init__(k=k)
        if not searxng_api_url:
            raise RuntimeError("Je moet searxng_api_url opgeven")
        self.searxng_api_url = searxng_api_url
        self.searxng_api_key = searxng_api_key
        self.usage = 0

        if is_valid_source:
            self.is_valid_source = is_valid_source
        else:
            self.is_valid_source = lambda x: True

    # Methode om het gebruik te resetten en te rapporteren
    def get_usage_and_reset(self):
        usage = self.usage
        self.usage = 0
        return {"SearXNG": usage}

    # Hoofdmethode voor het uitvoeren van zoekopdrachten
    def forward(
        self, query_or_queries: Union[str, List[str]], exclude_urls: List[str] = []
    ):
        """Zoek met SearXNG naar de top self.k passages voor query of queries

        Args:
            query_or_queries (Union[str, List[str]]): De query of queries om naar te zoeken.
            exclude_urls (List[str]): Een lijst met urls om uit te sluiten van de zoekresultaten.

        Returns:
            een lijst van Dicts, elke dict heeft sleutels 'description', 'snippets' (lijst van strings), 'title', 'url'
        """
        queries = (
            [query_or_queries]
            if isinstance(query_or_queries, str)
            else query_or_queries
        )
        self.usage += len(queries)
        collected_results = []
        headers = (
            {"Authorization": f"Bearer {self.searxng_api_key}"}
            if self.searxng_api_key
            else {}
        )

        for query in queries:
            try:
                params = {"q": query, "format": "json"}
                response = requests.get(
                    self.searxng_api_url, headers=headers, params=params
                )
                results = response.json()

                for r in results["results"]:
                    if self.is_valid_source(r["url"]) and r["url"] not in exclude_urls:
                        collected_results.append(
                            {
                                "description": r.get("content", ""),
                                "snippets": [r.get("content", "")],
                                "title": r.get("title", ""),
                                "url": r["url"],
                            }
                        )
            except Exception as e:
                logging.error(f"Fout bij het zoeken naar query {query}: {e}")

        return collected_results


class DuckDuckGoSearchRM(dspy.Retrieve):
    """Haal informatie op uit aangepaste zoekopdrachten met behulp van DuckDuckGo."""

    def __init__(
        self,
        k: int = 3,
        is_valid_source: Callable = None,
        min_char_count: int = 150,
        snippet_chunk_size: int = 1000,
        webpage_helper_max_threads=10,
        safe_search: str = "On",
        region: str = "us-en",
    ):
        """
        Initialiseert de DuckDuckGoSearchRM.

        Parameters:
            k: Aantal top resultaten om op te halen.
            is_valid_source: Optionele functie om geldige bronnen te filteren.
            min_char_count: Minimum aantal tekens voor een artikel om als geldig te worden beschouwd.
            snippet_chunk_size: Maximum aantal tekens voor elk snippet.
            webpage_helper_max_threads: Maximum aantal threads voor de webpage helper.
            safe_search: Instelling voor veilig zoeken ('On', 'Moderate', 'Off').
            region: Regio-instelling voor zoekresultaten (bijv. 'us-en', 'nl-nl').
        """
        super().__init__(k=k)
        try:
            from duckduckgo_search import DDGS
        except ImportError as err:
            raise ImportError(
                "Duckduckgo vereist `pip install duckduckgo_search`."
            ) from err
        self.k = k
        self.webpage_helper = WebPageHelper(
            min_char_count=min_char_count,
            snippet_chunk_size=snippet_chunk_size,
            max_thread_num=webpage_helper_max_threads,
        )
        self.usage = 0
        # Alle parameters voor zoeken kunnen hier worden gevonden:
        #   https://duckduckgo.com/duckduckgo-help-pages/settings/params/

        # Stelt de backend in op api
        self.duck_duck_go_backend = "api"

        # Haalt alleen veilige zoekresultaten op
        self.duck_duck_go_safe_search = safe_search

        # Specificeert de regio die de zoekopdracht zal gebruiken
        self.duck_duck_go_region = region

        # Als is_valid_source niet None is, moet het een functie zijn die een URL accepteert en een boolean teruggeeft.
        if is_valid_source:
            self.is_valid_source = is_valid_source
        else:
            self.is_valid_source = lambda x: True

        # Importeert de duckduckgo zoekbibliotheek gevonden op: https://github.com/deedy5/duckduckgo_search
        self.ddgs = DDGS()

    def get_usage_and_reset(self):
        """
        Haalt het huidige gebruik op en reset het naar nul.

        Returns:
            dict: Een dictionary met de sleutel "DuckDuckGoRM" en het huidige gebruik als waarde.
        """
        usage = self.usage
        self.usage = 0
        return {"DuckDuckGoRM": usage}

    @backoff.on_exception(
        backoff.expo,
        (Exception,),
        max_time=1000,
        max_tries=8,
        on_backoff=backoff_hdlr,
        giveup=giveup_hdlr,
    )
    def request(self, query: str):
        """
        Voert een zoekopdracht uit met DuckDuckGo.

        Args:
            query (str): De zoekopdracht.

        Returns:
            List[Dict]: Een lijst met zoekresultaten.
        """
        results = self.ddgs.text(
            query, max_results=self.k, backend=self.duck_duck_go_backend
        )
        return results

    def forward(
        self, query_or_queries: Union[str, List[str]], exclude_urls: List[str] = []
    ):
        """
        Zoekt met DuckDuckGoSearch naar de top self.k passages voor query of queries.

        Args:
            query_or_queries (Union[str, List[str]]): De query of queries om naar te zoeken.
            exclude_urls (List[str]): Een lijst met urls om uit te sluiten van de zoekresultaten.

        Returns:
            List[Dict]: Een lijst van dictionaries, elke dictionary heeft sleutels 'description', 'snippets' (lijst van strings), 'title', 'url'.
        """
        queries = (
            [query_or_queries]
            if isinstance(query_or_queries, str)
            else query_or_queries
        )
        self.usage += len(queries)

        collected_results = []

        for query in queries:
            # Lijst van dictionaries die zullen worden geparseerd om terug te geven
            results = self.request(query)

            for d in results:
                # Controleer of d een dictionary is
                if not isinstance(d, dict):
                    print(f"Ongeldig resultaat: {d}\n")
                    continue

                try:
                    # Zorg ervoor dat de sleutels aanwezig zijn
                    url = d.get("href", None)
                    title = d.get("title", None)
                    description = d.get("description", title)
                    snippets = [d.get("body", None)]

                    # Raise exception als er sleutel(s) ontbreken
                    if not all([url, title, description, snippets]):
                        raise ValueError(f"Ontbrekende sleutel(s) in resultaat: {d}")
                    if self.is_valid_source(url) and url not in exclude_urls:
                        result = {
                            "url": url,
                            "title": title,
                            "description": description,
                            "snippets": snippets,
                        }
                        collected_results.append(result)
                    else:
                        print(f"Ongeldige bron {url} of url in exclude_urls")
                except Exception as e:
                    print(f"Fout opgetreden bij het verwerken van {result=}: {e}\n")
                    print(f"Fout opgetreden bij het zoeken naar query {query}: {e}")

        return collected_results


class TavilySearchRM(dspy.Retrieve):
    """Haal informatie op uit aangepaste zoekopdrachten met behulp van Tavily. Documentatie en voorbeelden zijn te vinden op https://docs.tavily.com/docs/python-sdk/tavily-search/examples"""

    def __init__(
        self,
        tavily_search_api_key=None,
        k: int = 3,
        is_valid_source: Callable = None,
        min_char_count: int = 150,
        snippet_chunk_size: int = 1000,
        webpage_helper_max_threads=10,
        include_raw_content=False,
    ):
        """
        Initialiseert de TavilySearchRM.

        Parameters:
            tavily_search_api_key (str): API-sleutel voor Tavily die kan worden verkregen van https://tavily.com/
            k (int): Aantal top resultaten om op te halen.
            is_valid_source (Callable): Optionele functie om geldige bronnen te filteren.
            min_char_count (int): Minimum aantal tekens voor een artikel om als geldig te worden beschouwd.
            snippet_chunk_size (int): Maximum aantal tekens voor elk snippet.
            webpage_helper_max_threads (int): Maximum aantal threads voor de webpage helper.
            include_raw_content (bool): Boolean die wordt gebruikt om te bepalen of de volledige tekst moet worden geretourneerd.
        """
        super().__init__(k=k)
        try:
            from tavily import TavilyClient
        except ImportError as err:
            raise ImportError("Tavily vereist `pip install tavily-python`.") from err

        if not tavily_search_api_key and not os.environ.get("TAVILY_API_KEY"):
            raise RuntimeError(
                "Je moet tavily_search_api_key opgeven of de omgevingsvariabele TAVILY_API_KEY instellen"
            )
        elif tavily_search_api_key:
            self.tavily_search_api_key = tavily_search_api_key
        else:
            self.tavily_search_api_key = os.environ["TAVILY_API_KEY"]

        self.k = k
        self.webpage_helper = WebPageHelper(
            min_char_count=min_char_count,
            snippet_chunk_size=snippet_chunk_size,
            max_thread_num=webpage_helper_max_threads,
        )

        self.usage = 0

        # Maakt een client-instantie aan die zal worden gebruikt voor zoeken. Volledige zoekparameters zijn hier te vinden:
        # https://docs.tavily.com/docs/python-sdk/tavily-search/examples
        self.tavily_client = TavilyClient(api_key=self.tavily_search_api_key)

        self.include_raw_content = include_raw_content

        # Als is_valid_source niet None is, moet het een functie zijn die een URL accepteert en een boolean teruggeeft.
        if is_valid_source:
            self.is_valid_source = is_valid_source
        else:
            self.is_valid_source = lambda x: True

    def get_usage_and_reset(self):
        """
        Haalt het huidige gebruik op en reset het naar nul.

        Returns:
            dict: Een dictionary met de sleutel "TavilySearchRM" en het huidige gebruik als waarde.
        """
        usage = self.usage
        self.usage = 0
        return {"TavilySearchRM": usage}

    def forward(
        self, query_or_queries: Union[str, List[str]], exclude_urls: List[str] = []
    ):
        """
        Zoekt met TavilySearch naar de top self.k passages voor query of queries.

        Args:
            query_or_queries (Union[str, List[str]]): De query of queries om naar te zoeken.
            exclude_urls (List[str]): Een lijst met urls om uit te sluiten van de zoekresultaten.

        Returns:
            List[Dict]: Een lijst van dictionaries, elke dictionary heeft sleutels 'description', 'snippets' (lijst van strings), 'title', 'url'.
        """
        queries = (
            [query_or_queries]
            if isinstance(query_or_queries, str)
            else query_or_queries
        )
        self.usage += len(queries)

        collected_results = []

        for query in queries:
            args = {
                "max_results": self.k,
                "include_raw_contents": self.include_raw_content,
            }
            # Lijst van dictionaries die zullen worden geparseerd om terug te geven
            responseData = self.tavily_client.search(query)
            results = responseData.get("results")
            for d in results:
                # Controleer of d een dictionary is
                if not isinstance(d, dict):
                    print(f"Ongeldig resultaat: {d}\n")
                    continue

                try:
                    # Zorg ervoor dat de sleutels aanwezig zijn
                    url = d.get("url", None)
                    title = d.get("title", None)
                    description = d.get("content", None)
                    snippets = []
                    if d.get("raw_body_content"):
                        snippets.append(d.get("raw_body_content"))
                    else:
                        snippets.append(d.get("content"))

                    # Raise exception als er sleutel(s) ontbreken
                    if not all([url, title, description, snippets]):
                        raise ValueError(f"Ontbrekende sleutel(s) in resultaat: {d}")
                    if self.is_valid_source(url) and url not in exclude_urls:
                        result = {
                            "url": url,
                            "title": title,
                            "description": description,
                            "snippets": snippets,
                        }
                        collected_results.append(result)
                    else:
                        print(f"Ongeldige bron {url} of url in exclude_urls")
                except Exception as e:
                    print(f"Fout opgetreden bij het verwerken van {result=}: {e}\n")
                    print(f"Fout opgetreden bij het zoeken naar query {query}: {e}")

        return collected_results


class GoogleSearch(dspy.Retrieve):
    def __init__(
        self,
        google_search_api_key=None,
        google_cse_id=None,
        k=3,
        is_valid_source: Callable = None,
        min_char_count: int = 150,
        snippet_chunk_size: int = 1000,
        webpage_helper_max_threads=10,
    ):
        """
        Initialiseert de GoogleSearch klasse.

        Parameters:
            google_search_api_key: Google API-sleutel. Zie https://developers.google.com/custom-search/v1/overview
                sectie "API key"
            google_cse_id: Custom search engine ID. Zie https://developers.google.com/custom-search/v1/overview
                sectie "Search engine ID"
            k: Aantal top resultaten om op te halen.
            is_valid_source: Optionele functie om geldige bronnen te filteren.
            min_char_count: Minimum aantal tekens voor een artikel om als geldig te worden beschouwd.
            snippet_chunk_size: Maximum aantal tekens voor elk snippet.
            webpage_helper_max_threads: Maximum aantal threads voor de webpage helper.
        """
        super().__init__(k=k)
        try:
            from googleapiclient.discovery import build
        except ImportError as err:
            raise ImportError(
                "GoogleSearch vereist `pip install google-api-python-client`."
            ) from err
        if not google_search_api_key and not os.environ.get("GOOGLE_SEARCH_API_KEY"):
            raise RuntimeError(
                "Je moet google_search_api_key opgeven of de omgevingsvariabele GOOGLE_SEARCH_API_KEY instellen"
            )
        if not google_cse_id and not os.environ.get("GOOGLE_CSE_ID"):
            raise RuntimeError(
                "Je moet google_cse_id opgeven of de omgevingsvariabele GOOGLE_CSE_ID instellen"
            )

        self.google_search_api_key = (
            google_search_api_key or os.environ["GOOGLE_SEARCH_API_KEY"]
        )
        self.google_cse_id = google_cse_id or os.environ["GOOGLE_CSE_ID"]

        if is_valid_source:
            self.is_valid_source = is_valid_source
        else:
            self.is_valid_source = lambda x: True

        self.service = build(
            "customsearch", "v1", developerKey=self.google_search_api_key
        )
        self.webpage_helper = WebPageHelper(
            min_char_count=min_char_count,
            snippet_chunk_size=snippet_chunk_size,
            max_thread_num=webpage_helper_max_threads,
        )
        self.usage = 0

    def get_usage_and_reset(self):
        """
        Haalt het huidige gebruik op en reset het naar nul.

        Returns:
            dict: Een dictionary met de sleutel "GoogleSearch" en het huidige gebruik als waarde.
        """
        usage = self.usage
        self.usage = 0
        return {"GoogleSearch": usage}

    def forward(
        self, query_or_queries: Union[str, List[str]], exclude_urls: List[str] = []
    ):
        """
        Zoekt met de Google Custom Search API naar de top self.k resultaten voor query of queries.

        Args:
            query_or_queries (Union[str, List[str]]): De query of queries om naar te zoeken.
            exclude_urls (List[str]): Een lijst met URLs om uit te sluiten van de zoekresultaten.

        Returns:
            List[Dict]: Een lijst van dictionaries, elke dictionary heeft sleutels: 'title', 'url', 'snippet', 'description'.
        """
        queries = (
            [query_or_queries]
            if isinstance(query_or_queries, str)
            else query_or_queries
        )
        self.usage += len(queries)

        url_to_results = {}

        for query in queries:
            try:
                response = (
                    self.service.cse()
                    .list(
                        q=query,
                        cx=self.google_cse_id,
                        num=self.k,
                    )
                    .execute()
                )

                for item in response.get("items", []):
                    if (
                        self.is_valid_source(item["link"])
                        and item["link"] not in exclude_urls
                    ):
                        url_to_results[item["link"]] = {
                            "title": item["title"],
                            "url": item["link"],
                            # "snippet": item.get("snippet", ""),  # Google zoeksnippet is erg kort.
                            "description": item.get("snippet", ""),
                        }

            except Exception as e:
                logging.error(f"Fout opgetreden bij het zoeken naar query {query}: {e}")

        valid_url_to_snippets = self.webpage_helper.urls_to_snippets(
            list(url_to_results.keys())
        )
        collected_results = []
        for url in valid_url_to_snippets:
            r = url_to_results[url]
            r["snippets"] = valid_url_to_snippets[url]["snippets"]
            collected_results.append(r)

        return collected_results


class AzureAISearch(dspy.Retrieve):
    """Retrieve information from custom queries using Azure AI Search.

    General Documentation: https://learn.microsoft.com/en-us/azure/search/search-create-service-portal.
    Python Documentation: https://learn.microsoft.com/en-us/python/api/overview/azure/search-documents-readme?view=azure-python.
    """

    def __init__(
        self,
        azure_ai_search_api_key=None,
        azure_ai_search_url=None,
        azure_ai_search_index_name=None,
        k=3,
        is_valid_source: Callable = None,
    ):
        """
        Params:
            azure_ai_search_api_key: Azure AI Search API key. Check out https://learn.microsoft.com/en-us/azure/search/search-security-api-keys?tabs=rest-use%2Cportal-find%2Cportal-query
                "API key" section
            azure_ai_search_url: Custom Azure AI Search Endpoint URL. Check out https://learn.microsoft.com/en-us/azure/search/search-create-service-portal#name-the-service
            azure_ai_search_index_name: Custom Azure AI Search Index Name. Check out https://learn.microsoft.com/en-us/azure/search/search-how-to-create-search-index?tabs=portal
            k: Number of top results to retrieve.
            is_valid_source: Optional function to filter valid sources.
            min_char_count: Minimum character count for the article to be considered valid.
            snippet_chunk_size: Maximum character count for each snippet.
            webpage_helper_max_threads: Maximum number of threads to use for webpage helper.
        """
        super().__init__(k=k)

        try:
            from azure.core.credentials import AzureKeyCredential
            from azure.search.documents import SearchClient
        except ImportError as err:
            raise ImportError(
                "AzureAISearch requires `pip install azure-search-documents`."
            ) from err

        if not azure_ai_search_api_key and not os.environ.get(
            "AZURE_AI_SEARCH_API_KEY"
        ):
            raise RuntimeError(
                "You must supply azure_ai_search_api_key or set environment variable AZURE_AI_SEARCH_API_KEY"
            )
        elif azure_ai_search_api_key:
            self.azure_ai_search_api_key = azure_ai_search_api_key
        else:
            self.azure_ai_search_api_key = os.environ["AZURE_AI_SEARCH_API_KEY"]

        if not azure_ai_search_url and not os.environ.get("AZURE_AI_SEARCH_URL"):
            raise RuntimeError(
                "You must supply azure_ai_search_url or set environment variable AZURE_AI_SEARCH_URL"
            )
        elif azure_ai_search_url:
            self.azure_ai_search_url = azure_ai_search_url
        else:
            self.azure_ai_search_url = os.environ["AZURE_AI_SEARCH_URL"]

        if not azure_ai_search_index_name and not os.environ.get(
            "AZURE_AI_SEARCH_INDEX_NAME"
        ):
            raise RuntimeError(
                "You must supply azure_ai_search_index_name or set environment variable AZURE_AI_SEARCH_INDEX_NAME"
            )
        elif azure_ai_search_index_name:
            self.azure_ai_search_index_name = azure_ai_search_index_name
        else:
            self.azure_ai_search_index_name = os.environ["AZURE_AI_SEARCH_INDEX_NAME"]

        self.usage = 0

        # If not None, is_valid_source shall be a function that takes a URL and returns a boolean.
        if is_valid_source:
            self.is_valid_source = is_valid_source
        else:
            self.is_valid_source = lambda x: True

    def get_usage_and_reset(self):
        usage = self.usage
        self.usage = 0

        return {"AzureAISearch": usage}

    def forward(
        self, query_or_queries: Union[str, List[str]], exclude_urls: List[str] = []
    ):
        """Search with Azure Open AI for self.k top passages for query or queries

        Args:
            query_or_queries (Union[str, List[str]]): The query or queries to search for.
            exclude_urls (List[str]): A list of urls to exclude from the search results.

        Returns:
            a list of Dicts, each dict has keys of 'description', 'snippets' (list of strings), 'title', 'url'
        """
        try:
            from azure.core.credentials import AzureKeyCredential
            from azure.search.documents import SearchClient
        except ImportError as err:
            raise ImportError(
                "AzureAISearch requires `pip install azure-search-documents`."
            ) from err
        queries = (
            [query_or_queries]
            if isinstance(query_or_queries, str)
            else query_or_queries
        )
        self.usage += len(queries)
        collected_results = []

        client = SearchClient(
            self.azure_ai_search_url,
            self.azure_ai_search_index_name,
            AzureKeyCredential(self.azure_ai_search_api_key),
        )
        for query in queries:
            try:
                # https://learn.microsoft.com/en-us/python/api/azure-search-documents/azure.search.documents.searchclient?view=azure-python#azure-search-documents-searchclient-search
                results = client.search(search_text=query, top=1)

                for result in results:
                    document = {
                        "url": result["metadata_storage_path"],
                        "title": result["title"],
                        "description": "N/A",
                        "snippets": [result["chunk"]],
                    }
                    collected_results.append(document)
            except Exception as e:
                logging.error(f"Error occurs when searching query {query}: {e}")

        return collected_results
