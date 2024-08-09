from typing import Union, List, Dict, Any
from urllib.parse import urlparse
import logging

import dspy
from knowledge_storm.base import Retriever
from knowledge_storm.rm import YouRM, VectorRM
from ...utils import ArticleTextProcessing

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Internet source restrictions according to Wikipedia standard:
# https://en.wikipedia.org/wiki/Wikipedia:Reliable_sources/Perennial_sources
GENERALLY_UNRELIABLE = {
    "112_Ukraine",
    "Ad_Fontes_Media",
    "AlterNet",
    "Amazon",
    "Anadolu_Agency_(controversial_topics)",
    "Ancestry.com",
    "Answers.com",
    "Antiwar.com",
    "Anti-Defamation_League",
    "arXiv",
    "Atlas_Obscura_places",
    "Bild",
    "Blaze_Media",
    "Blogger",
    "BroadwayWorld",
    "California_Globe",
    "The_Canary",
    "CelebrityNetWorth",
    "CESNUR",
    "ChatGPT",
    "CNET_(November_2022\u2013present)",
    "CoinDesk",
    "Consortium_News",
    "CounterPunch",
    "Correo_del_Orinoco",
    "Cracked.com",
    "Daily_Express",
    "Daily_Kos",
    "Daily_Sabah",
    "The_Daily_Wire",
    "Discogs",
    "Distractify",
    "The_Electronic_Intifada",
    "Encyclopaedia_Metallum",
    "Ethnicity_of_Celebs",
    "Facebook",
    "FamilySearch",
    "Fandom",
    "The_Federalist",
    "Find_a_Grave",
    "Findmypast",
    "Flags_of_the_World",
    "Flickr",
    "Forbes.com_contributors",
    "Fox_News_(politics_and_science)",
    "Fox_News_(talk_shows)",
    "Gawker",
    "GB_News",
    "Geni.com",
    "gnis-class",
    "gns-class",
    "GlobalSecurity.org",
    "Goodreads",
    "Guido_Fawkes",
    "Heat_Street",
    "History",
    "HuffPost_contributors",
    "IMDb",
    "Independent_Media_Center",
    "Inquisitr",
    "International_Business_Times",
    "Investopedia",
    "Jewish_Virtual_Library",
    "Joshua_Project",
    "Know_Your_Meme",
    "Land_Transport_Guru",
    "LinkedIn",
    "LiveJournal",
    "Marquis_Who's_Who",
    "Mashable_sponsored_content",
    "MEAWW",
    "Media_Bias/Fact_Check",
    "Media_Research_Center",
    "Medium",
    "metal-experience",
    "Metro",
    "The_New_American",
    "New_York_Post",
    "NGO_Monitor",
    "The_Onion",
    "Our_Campaigns",
    "PanAm_Post",
    "Patheos",
    "An_Phoblacht",
    "The_Post_Millennial",
    "arXiv",
    "bioRxiv",
    "medRxiv",
    "PeerJ Preprints",
    "Preprints.org",
    "SSRN",
    "PR_Newswire",
    "Quadrant",
    "Quillette",
    "Quora",
    "Raw_Story",
    "Reddit",
    "RedState",
    "ResearchGate",
    "Rolling_Stone_(politics_and_society,_2011\u2013present)",
    "Rolling_Stone_(Culture_Council)",
    "Scribd",
    "Scriptural_texts",
    "Simple_Flying",
    "Sixth_Tone_(politics)",
    "The_Skwawkbox",
    "SourceWatch",
    "Spirit_of_Metal",
    "Sportskeeda",
    "Stack_Exchange",
    "Stack_Overflow",
    "MathOverflow",
    "Ask_Ubuntu",
    "starsunfolded.com",
    "Statista",
    "TASS",
    "The_Truth_About_Guns",
    "TV.com",
    "TV_Tropes",
    "Twitter",
    "X.com",
    "Urban_Dictionary",
    "Venezuelanalysis",
    "VGChartz",
    "VoC",
    "Washington_Free_Beacon",
    "Weather2Travel",
    "The_Western_Journal",
    "We_Got_This_Covered",
    "WhatCulture",
    "Who's_Who_(UK)",
    "WhoSampled",
    "Wikidata",
    "WikiLeaks",
    "Wikinews",
    "Wikipedia",
    "WordPress.com",
    "Worldometer",
    "YouTube",
    "ZDNet"}
DEPRECATED = {
    "Al_Mayadeen",
    "ANNA_News",
    "Baidu_Baike",
    "China_Global_Television_Network",
    "The_Cradle",
    "Crunchbase",
    "The_Daily_Caller",
    "Daily_Mail",
    "Daily_Star",
    "The_Epoch_Times",
    "FrontPage_Magazine",
    "The_Gateway_Pundit",
    "Global_Times",
    "The_Grayzone",
    "HispanTV",
    "Jihad_Watch",
    "Last.fm",
    "LifeSiteNews",
    "The_Mail_on_Sunday",
    "MintPress_News",
    "National_Enquirer",
    "New_Eastern_Outlook",
    "News_Break",
    "NewsBlaze",
    "News_of_the_World",
    "Newsmax",
    "NNDB",
    "Occupy_Democrats",
    "Office_of_Cuba_Broadcasting",
    "One_America_News_Network",
    "Peerage_websites",
    "Press_TV",
    "Project_Veritas",
    "Rate_Your_Music",
    "Republic_TV",
    "Royal_Central",
    "RT",
    "Sputnik",
    "The_Sun",
    "Taki's_Magazine",
    "Tasnim_News_Agency",
    "Telesur",
    "The_Unz_Review",
    "VDARE",
    "Voltaire_Network",
    "WorldNetDaily",
    "Zero_Hedge"
}
BLACKLISTED = {
    "Advameg",
    "bestgore.com",
    "Breitbart_News",
    "Centre_for_Research_on_Globalization",
    "Examiner.com",
    "Famous_Birthdays",
    "Healthline",
    "InfoWars",
    "Lenta.ru",
    "LiveLeak",
    "Lulu.com",
    "MyLife",
    "Natural_News",
    "OpIndia",
    "The_Points_Guy",
    "The_Points_Guy_(sponsored_content)",
    "Swarajya",
    "Veterans_Today",
    "ZoomInfo"
}

def is_valid_wikipedia_source(url):
    parsed_url = urlparse(url)
    combined_set = GENERALLY_UNRELIABLE | DEPRECATED | BLACKLISTED
    for domain in combined_set:
        if domain in parsed_url.netloc:
            logger.debug(f"URL {url} is niet geldig vanwege domein {domain}")
            return False
    logger.debug(f"URL {url} is geldig")
    return True

class StormRetriever(Retriever):
    def __init__(self, rm: dspy.Retrieve, k=3):
        super().__init__(search_top_k=k)
        self._rm = rm
        if hasattr(rm, 'is_valid_source'):
            rm.is_valid_source = is_valid_wikipedia_source
        logger.info(f"StormRetriever geïnitialiseerd met k: {k}")

    def retrieve(self, query: Union[str, List[str]], exclude_urls: List[str] = []) -> List[Any]:
        from knowledge_storm.storm_wiki.modules.storm_dataclass import StormInformation
        
        logger.info(f"StormRetriever.retrieve aangeroepen met query: {query}, exclude_urls: {exclude_urls}")
        retrieved_data_list = self._rm(query_or_queries=query, exclude_urls=exclude_urls)
        processed_results = []
        for data in retrieved_data_list:
            if 'snippets' in data:
                data['snippets'] = [ArticleTextProcessing.remove_citations(snippet) for snippet in data['snippets']]
            processed_results.append(StormInformation.from_dict(data))
        logger.info(f"Aantal verwerkte resultaten: {len(processed_results)}")
        return processed_results

    def update_search_top_k(self, k: int):
        self.search_top_k = k
        if hasattr(self._rm, 'update_search_top_k'):
            self._rm.update_search_top_k(k)
        logger.info(f"StormRetriever search_top_k updated to: {k}")

    def collect_and_reset_rm_usage(self):
        return self._rm.get_usage_and_reset() if hasattr(self._rm, 'get_usage_and_reset') else {}

class VectorRMRetriever(Retriever):
    def __init__(self, vector_rm: VectorRM, k: int = 3):
        super().__init__(search_top_k=k)
        self.vector_rm = vector_rm
        logger.info(f"VectorRMRetriever geïnitialiseerd met k: {k}")

    def retrieve(self, query: Union[str, List[str]], exclude_urls: List[str] = []) -> List[Any]:
        from knowledge_storm.storm_wiki.modules.storm_dataclass import StormInformation
        
        logger.info(f"VectorRMRetriever.retrieve aangeroepen met query: {query}")
        
        results = self.vector_rm.retrieve(query)
        
        processed_results = []
        for result in results:
            processed_results.append(StormInformation(
                snippets=[result['page_content']],
                score=float(result['score']),
                url="",
                title=f"Document: {result['metadata'].get('source', 'Onbekende bron')}"
            ))
        
        logger.info(f"Aantal verwerkte VectorRM resultaten: {len(processed_results)}")
        return processed_results[:self.search_top_k]

    def update_search_top_k(self, k: int):
        self.search_top_k = k
        if hasattr(self.vector_rm, 'update_search_top_k'):
            self.vector_rm.update_search_top_k(k)
        logger.info(f"VectorRMRetriever search_top_k updated to: {k}")

    def collect_and_reset_rm_usage(self):
        return self.vector_rm.get_usage_and_reset() if hasattr(self.vector_rm, 'get_usage_and_reset') else {}

    def add_directory(self, directory_path: str):
        self.vector_rm.add_directory(directory_path)

    def add_file(self, file_path: str):
        self.vector_rm.add_file(file_path)

class CombinedRetriever(Retriever):
    def __init__(self, storm_retriever: StormRetriever, vector_retriever: VectorRMRetriever, search_top_k: int):
        super().__init__(search_top_k=search_top_k)
        self.storm_retriever = storm_retriever
        self.vector_retriever = vector_retriever
        logger.info(f"CombinedRetriever geïnitialiseerd met search_top_k: {search_top_k}")

    def retrieve(self, query: Union[str, List[str]], exclude_urls: List[str] = []) -> List[Any]:
        storm_results = self.storm_retriever.retrieve(query, exclude_urls=exclude_urls)
        
        if len(storm_results) < self.search_top_k:
            remaining_k = self.search_top_k - len(storm_results)
            vector_results = self.vector_retriever.retrieve(query)[:remaining_k]
            combined_results = storm_results + vector_results
        else:
            combined_results = storm_results
        
        return combined_results[:self.search_top_k]

    def update_search_top_k(self, k: int):
        self.search_top_k = k
        self.storm_retriever.update_search_top_k(k)
        self.vector_retriever.update_search_top_k(k)
        logger.info(f"CombinedRetriever search_top_k updated to: {k}")

    def collect_and_reset_rm_usage(self):
        storm_usage = self.storm_retriever.collect_and_reset_rm_usage()
        vector_usage = self.vector_retriever.collect_and_reset_rm_usage()
        return {**storm_usage, **vector_usage}

    def add_directory(self, directory_path: str):
        self.vector_retriever.add_directory(directory_path)

    def add_file(self, file_path: str):
        self.vector_retriever.add_file(file_path)