import concurrent.futures
import dspy
import functools
import hashlib
import json
import logging
import time
from abc import ABC, abstractmethod
from collections import OrderedDict
from typing import Dict, List, Optional, Union, TYPE_CHECKING

from .utils import ArticleTextProcessing

# Configureer logging
logging.basicConfig(
    level=logging.INFO, format="%(name)s : %(levelname)-8s : %(message)s"
)
logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from .logging_wrapper import LoggingWrapper


class InformationTable(ABC):
    """
    De InformationTable klasse dient als dataklasse om informatie op te slaan
    die verzameld is tijdens de KnowledgeCuration fase.

    Maak een subklasse om indien nodig meer informatie op te nemen. Bijvoorbeeld,
    in het STORM-paper (https://arxiv.org/pdf/2402.14207.pdf) zou aanvullende informatie
    de perspectiefgestuurde dialooggeschiedenis zijn.
    """

    def __init__(self):
        pass

    @abstractmethod
    def retrieve_information(**kwargs):
        pass


class Information:
    """Klasse om gedetailleerde informatie weer te geven.

    Erft over van Information om een unieke identifier (URL) op te nemen, en breidt
    dit uit met een beschrijving, snippets en titel van de storm-informatie.

    Attributen:
        description (str): Korte beschrijving.
        snippets (list): Lijst van korte uittreksels of fragmenten.
        title (str): De titel of kop van de informatie.
        url (str): De unieke URL (dienend als UUID) van de informatie.
    """

    def __init__(self, url, description, snippets, title, meta=None):
        """Initialiseer het Information object met gedetailleerde attributen.

        Args:
            url (str): De unieke URL die dient als identifier voor de informatie.
            description (str): Gedetailleerde beschrijving.
            snippets (list): Lijst van korte uittreksels of fragmenten.
            title (str): De titel of kop van de informatie.
        """
        self.description = description
        self.snippets = snippets
        self.title = title
        self.url = url
        self.meta = meta if meta is not None else {}
        self.citation_uuid = -1

    def __hash__(self):
        return hash(
            (
                self.url,
                tuple(sorted(self.snippets)),
            )
        )

    def __eq__(self, other):
        if not isinstance(other, Information):
            return False
        return (
            self.url == other.url
            and set(self.snippets) == set(other.snippets)
            and self._meta_str() == other._meta_str()
        )

    def __hash__(self):
        return int(
            self._md5_hash((self.url, tuple(sorted(self.snippets)), self._meta_str())),
            16,
        )

    def _meta_str(self):
        """Genereer een string-representatie van relevante meta-informatie."""
        return f"Vraag: {self.meta.get('question', '')}, Zoekopdracht: {self.meta.get('query', '')}"

    def _md5_hash(self, value):
        """Genereer een MD5-hash voor een gegeven waarde."""
        if isinstance(value, (dict, list, tuple)):
            value = json.dumps(value, sort_keys=True)
        return hashlib.md5(str(value).encode("utf-8")).hexdigest()

    @classmethod
    def from_dict(cls, info_dict):
        """Maak een Information object van een dictionary.
           Gebruik: info = Information.from_dict(storm_info_dict)

        Args:
            info_dict (dict): Een dictionary met keys 'url', 'description',
                              'snippets', en 'title' die overeenkomen met de attributen van het object.

        Returns:
            Information: Een instantie van Information.
        """
        info = cls(
            url=info_dict["url"],
            description=info_dict["description"],
            snippets=info_dict["snippets"],
            title=info_dict["title"],
            meta=info_dict.get("meta", None),
        )
        info.citation_uuid = int(info_dict.get("citation_uuid", -1))
        return info

    def to_dict(self):
        return {
            "url": self.url,
            "description": self.description,
            "snippets": self.snippets,
            "title": self.title,
            "meta": self.meta,
            "citation_uuid": self.citation_uuid,
        }


class ArticleSectionNode:
    """
    De ArticleSectionNode is de dataklasse voor het verwerken van de sectie van het artikel.
    De opslag van inhoud en schrijfvoorkeuren voor secties worden in deze node gedefinieerd.
    """

    def __init__(self, section_name: str, content=None):
        """
        section_name: sectiekop in string-formaat. Bijv. Inleiding, Geschiedenis, etc.
        content: inhoud van de sectie. De keuze voor de datastructuur is aan jou.
        """
        self.section_name = section_name
        self.content = content
        self.children = []
        self.preference = None

    def add_child(self, new_child_node, insert_to_front=False):
        if insert_to_front:
            self.children.insert(0, new_child_node)
        else:
            self.children.append(new_child_node)

    def remove_child(self, child):
        self.children.remove(child)


class Article(ABC):
    def __init__(self, topic_name):
        self.root = ArticleSectionNode(topic_name)

    def find_section(
        self, node: ArticleSectionNode, name: str
    ) -> Optional[ArticleSectionNode]:
        """
        Geef de node van de sectie terug op basis van de sectienaam.

        Args:
            node: de node als root om te zoeken.
            name: de naam van de node als sectienaam

        Return:
            referentie van de node of None als de sectienaam geen overeenkomst heeft
        """
        if node.section_name == name:
            return node
        for child in node.children:
            result = self.find_section(child, name)
            if result:
                return result
        return None

    @abstractmethod
    def to_string(self) -> str:
        """
        Exporteer Article object naar string-representatie.
        """

    def get_outline_tree(self):
        """
        Genereert een hiërarchische boomstructuur die de outline van het document weergeeft.

        Returns:
            Dict[str, Dict]: Een geneste dictionary die de hiërarchische structuur van de document-outline weergeeft.
                             Elke sleutel is een sectienaam, en de waarde is een andere dictionary die de onderliggende secties vertegenwoordigt,
                             recursief de boomstructuur van de document-outline vormend. Als een sectie geen subsecties heeft,
                             is de waarde een lege dictionary.

        Voorbeeld:
            Uitgaande van een document met een structuur zoals:
            - Inleiding
                - Achtergrond
                - Doelstelling
            - Methoden
                - Gegevensverzameling
                - Analyse
            De methode zou het volgende teruggeven:
            {
                'Inleiding': {
                    'Achtergrond': {},
                    'Doelstelling': {}
                },
                'Methoden': {
                    'Gegevensverzameling': {},
                    'Analyse': {}
                }
            }
        """

        def build_tree(node) -> Dict[str, Dict]:
            tree = {}
            for child in node.children:
                tree[child.section_name] = build_tree(child)
            return tree if tree else {}

        return build_tree(self.root)

    def get_first_level_section_names(self) -> List[str]:
        """
        Haal de namen van de secties op het eerste niveau op
        """
        return [i.section_name for i in self.root.children]

    @classmethod
    @abstractmethod
    def from_string(cls, topic_name: str, article_text: str):
        """
        Maak een instantie van het Article object van een string
        """
        pass

    def prune_empty_nodes(self, node=None):
        if node is None:
            node = self.root

        node.children[:] = [
            child for child in node.children if self.prune_empty_nodes(child)
        ]

        if (node.content is None or node.content == "") and not node.children:
            return None
        else:
            return node


class Retriever:
    """
    Een abstracte basisklasse voor retriever modules. Het biedt een sjabloon voor het ophalen van informatie op basis van een query.

    Deze klasse moet worden uitgebreid om specifieke ophaalmogelijkheden te implementeren.
    Gebruikers kunnen hun retriever modules naar behoefte ontwerpen door de retrieve-methode te implementeren.
    Het ophaalmodel/de zoekmachine die voor elk onderdeel wordt gebruikt, moet worden gedeclareerd met een achtervoegsel '_rm' in de attribuutnaam.
    """

    def __init__(self, rm: dspy.Retrieve, max_thread: int = 1):
        self.max_thread = max_thread
        self.rm = rm

    def collect_and_reset_rm_usage(self):
        combined_usage = []
        if hasattr(getattr(self, "rm"), "get_usage_and_reset"):
            combined_usage.append(getattr(self, "rm").get_usage_and_reset())

        name_to_usage = {}
        for usage in combined_usage:
            for model_name, query_cnt in usage.items():
                if model_name not in name_to_usage:
                    name_to_usage[model_name] = query_cnt
                else:
                    name_to_usage[model_name] += query_cnt

        return name_to_usage

    def retrieve(
        self, query: Union[str, List[str]], exclude_urls: List[str] = []
    ) -> List[Information]:
        queries = query if isinstance(query, list) else [query]
        to_return = []

        def process_query(q):
            retrieved_data_list = self.rm(
                query_or_queries=[q], exclude_urls=exclude_urls
            )
            local_to_return = []
            for data in retrieved_data_list:
                for i in range(len(data["snippets"])):
                    # STORM genereert het artikel met citaten. We beschouwen geen multi-hop citaten.
                    # Verwijder citaten in de bron om verwarring te voorkomen.
                    data["snippets"][i] = ArticleTextProcessing.remove_citations(
                        data["snippets"][i]
                    )
                storm_info = Information.from_dict(data)
                storm_info.meta["query"] = q
                local_to_return.append(storm_info)
            return local_to_return

        with concurrent.futures.ThreadPoolExecutor(
            max_workers=self.max_thread
        ) as executor:
            results = list(executor.map(process_query, queries))

        for result in results:
            to_return.extend(result)

        return to_return


class KnowledgeCurationModule(ABC):
    """
    De interface voor de kenniscuratiefase. Gegeven een onderwerp, retourneert verzamelde informatie.
    """

    def __init__(self, retriever: Retriever):
        """
        Sla argumenten op en voltooi initialisatie.
        """
        self.retriever = retriever

    @abstractmethod
    def research(self, topic) -> InformationTable:
        """
        Cureer informatie en kennis voor het gegeven onderwerp

        Args:
            topic: onderwerp van interesse in natuurlijke taal.

        Returns:
            collected_information: verzamelde informatie in InformationTable type.
        """
        pass


class OutlineGenerationModule(ABC):
    """
    De interface voor de outline-generatiefase. Gegeven onderwerp, verzamelde informatie uit de
    kenniscuratiefase, genereer een outline voor het artikel.
    """

    @abstractmethod
    def generate_outline(
        self, topic: str, information_table: InformationTable, **kwargs
    ) -> Article:
        """
        Genereer outline voor het artikel. Vereiste argumenten zijn:
            topic: het onderwerp van interesse
            information_table: kenniscuratiegegevens gegenereerd door KnowledgeCurationModule

        Meer argumenten kunnen zijn
            1. concept outline
            2. door gebruiker verstrekte outline

        Returns:
            article_outline van het type ArticleOutline
        """
        pass


class ArticleGenerationModule(ABC):
    """
    De interface voor de artikelgeneratiefase. Gegeven onderwerp, verzamelde informatie uit
    kenniscuratiefase, gegenereerde outline uit outline-generatiefase,
    """

    @abstractmethod
    def generate_article(
        self,
        topic: str,
        information_table: InformationTable,
        article_with_outline: Article,
        **kwargs,
    ) -> Article:
        """
        Genereer artikel. Vereiste argumenten zijn:
            topic: het onderwerp van interesse
            information_table: kenniscuratiegegevens gegenereerd door KnowledgeCurationModule
            article_with_outline: artikel met gespecificeerde outline van OutlineGenerationModule
        """
        pass


class ArticlePolishingModule(ABC):
    """
    De interface voor de artikelpolijstfase. Gegeven onderwerp, verzamelde informatie uit
    kenniscuratiefase, gegenereerde outline uit outline-generatiefase,
    """

    @abstractmethod
    def polish_article(self, topic: str, draft_article: Article, **kwargs) -> Article:
        """
        Polijst artikel. Vereiste argumenten zijn:
            topic: het onderwerp van interesse
            draft_article: conceptartikel van ArticleGenerationModule.
        """
        pass


def log_execution_time(func):
    """Decorator om de uitvoeringstijd van een functie te loggen."""

    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        start_time = time.time()
        result = func(self, *args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time
        logger.info(f"{func.__name__} uitgevoerd in {execution_time:.4f} seconden")
        self.time[func.__name__] = execution_time
        return result

    return wrapper


class LMConfigs(ABC):
    """Abstracte basisklasse voor taalmodelconfiguraties van de kenniscuratie-engine.

    Het taalmodel dat voor elk onderdeel wordt gebruikt, moet worden gedeclareerd met een achtervoegsel '_lm' in de attribuutnaam.
    """

    def __init__(self):
        pass

    def init_check(self):
        for attr_name in self.__dict__:
            if "_lm" in attr_name and getattr(self, attr_name) is None:
                logging.warning(
                    f"Taalmodel voor {attr_name} is niet geïnitialiseerd. Roep set_{attr_name}() aan"
                )

    def collect_and_reset_lm_history(self):
        history = []
        for attr_name in self.__dict__:
            if "_lm" in attr_name and hasattr(getattr(self, attr_name), "history"):
                history.extend(getattr(self, attr_name).history)
                getattr(self, attr_name).history = []

        return history

    def collect_and_reset_lm_usage(self):
        combined_usage = []
        for attr_name in self.__dict__:
            if "_lm" in attr_name and hasattr(
                getattr(self, attr_name), "get_usage_and_reset"
            ):
                combined_usage.append(getattr(self, attr_name).get_usage_and_reset())

        model_name_to_usage = {}
        for usage in combined_usage:
            for model_name, tokens in usage.items():
                if model_name not in model_name_to_usage:
                    model_name_to_usage[model_name] = tokens
                else:
                    model_name_to_usage[model_name]["prompt_tokens"] += tokens[
                        "prompt_tokens"
                    ]
                    model_name_to_usage[model_name]["completion_tokens"] += tokens[
                        "completion_tokens"
                    ]

        return model_name_to_usage

    def log(self):

        return OrderedDict(
            {
                attr_name: getattr(self, attr_name).kwargs
                for attr_name in self.__dict__
                if "_lm" in attr_name and hasattr(getattr(self, attr_name), "kwargs")
            }
        )


class Engine(ABC):
    def __init__(self, lm_configs: LMConfigs):
        self.lm_configs = lm_configs
        self.time = {}
        self.lm_cost = {}  # Kosten van taalmodellen gemeten in in/out tokens.
        self.rm_cost = {}  # Kosten van retrievers gemeten in aantal queries.

    def log_execution_time_and_lm_rm_usage(self, func):
        """Decorator om de uitvoeringstijd, taalmodelgebruik en retrieval modelgebruik van een functie te loggen."""

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            result = func(*args, **kwargs)
            end_time = time.time()
            execution_time = end_time - start_time
            self.time[func.__name__] = execution_time
            logger.info(f"{func.__name__} uitgevoerd in {execution_time:.4f} seconden")
            self.lm_cost[func.__name__] = self.lm_configs.collect_and_reset_lm_usage()
            if hasattr(self, "retriever"):
                self.rm_cost[func.__name__] = (
                    self.retriever.collect_and_reset_rm_usage()
                )
            return result

        return wrapper

    def apply_decorators(self):
        """Pas decorators toe op methoden die ze nodig hebben."""
        methods_to_decorate = [
            method_name
            for method_name in dir(self)
            if callable(getattr(self, method_name)) and method_name.startswith("run_")
        ]
        for method_name in methods_to_decorate:
            original_method = getattr(self, method_name)
            decorated_method = self.log_execution_time_and_lm_rm_usage(original_method)
            setattr(self, method_name, decorated_method)

    @abstractmethod
    def run_knowledge_curation_module(self, **kwargs) -> Optional[InformationTable]:
        pass

    @abstractmethod
    def run_outline_generation_module(self, **kwarg) -> Article:
        pass

    @abstractmethod
    def run_article_generation_module(self, **kwarg) -> Article:
        pass

    @abstractmethod
    def run_article_polishing_module(self, **kwarg) -> Article:
        pass

    @abstractmethod
    def run(self, **kwargs):
        pass

    def summary(self):
        print("***** Uitvoeringstijd *****")
        for k, v in self.time.items():
            print(f"{k}: {v:.4f} seconden")

        print("***** Tokengebruik van taalmodellen: *****")
        for k, v in self.lm_cost.items():
            print(f"{k}")
            for model_name, tokens in v.items():
                print(f"    {model_name}: {tokens}")

        print("***** Aantal queries van retrieval modellen: *****")
        for k, v in self.rm_cost.items():
            print(f"{k}: {v}")

    def reset(self):
        self.time = {}
        self.lm_cost = {}
        self.rm_cost = {}


class Agent(ABC):
    """
    Interface voor STORM en Co-STORM LLM agent

    Deze klasse moet worden geïmplementeerd door elke subklasse van `Agent` om te definiëren hoe de agent een uiting genereert.
    De gegenereerde uiting kan worden beïnvloed door de gespreksgeschiedenis, kennisbank en eventuele aanvullende parameters die via `kwargs` worden doorgegeven.
    De implementatie moet aansluiten bij de specifieke rol en het perspectief van de agent, zoals gedefinieerd door het onderwerp, de rolnaam en de rolbeschrijving van de agent.

    Args:
        knowledge_base (KnowledgeBase): De huidige kennisbank (bijv. mindmap in Co-STORM) die de verzamelde informatie bevat die relevant is voor het gesprek.
        conversation_history (List[ConversationTurn]): Een lijst van eerdere gespreksbeurten, die context biedt voor het genereren van de volgende uiting.
                                                       De agent kan naar deze geschiedenis verwijzen om continuïteit en relevantie in het gesprek te behouden.
        logging_wrapper (LoggingWrapper): Een wrapper die wordt gebruikt voor het loggen van belangrijke gebeurtenissen tijdens het genereren van uitingen.
        **kwargs: Aanvullende argumenten die aan de methode kunnen worden doorgegeven voor meer gespecialiseerd gedrag bij het genereren van uitingen, afhankelijk van de specifieke implementatie van de agent.

    Returns:
        ConversationTurn: Een nieuwe gespreksbeurt gegenereerd door de agent, met daarin de respons van de agent, inclusief de rol, het type uiting en relevante informatie uit de kennisbank.

    Opmerkingen:
        - Subklassen van `Agent` moeten de exacte strategie definiëren voor het genereren van de uiting, wat kanxs inhouden dat er wordt geïnteracteerd met een taalmodel, relevante kennis wordt opgehaald of specifieke gespreksregels worden gevolgd.
        - De rol van de agent, het perspectief en de inhoud van de kennisbank zullen invloed hebben op hoe de uiting wordt geformuleerd.
    """

    from .dataclass import KnowledgeBase, ConversationTurn

    def __init__(self, topic: str, role_name: str, role_description: str):
        self.topic = topic
        self.role_name = role_name
        self.role_description = role_description

    def get_role_description(self):
        if self.role_description:
            return f"{self.role_name}: {self.role_description}"
        return self.role_name

    @abstractmethod
    def generate_utterance(
        self,
        knowledge_base: KnowledgeBase,
        conversation_history: List[ConversationTurn],
        logging_wrapper: "LoggingWrapper",
        **kwargs,
    ):
        pass