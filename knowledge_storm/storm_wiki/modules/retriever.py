from typing import Union, List, Dict
from urllib.parse import urlparse
import logging

import dspy

from knowledge_storm.rm import YouRM, BingSearch, VectorRM
from .storm_dataclass import StormInformation
from ...interface import Retriever, Information
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
            return False
    return True

from typing import Union, List, Dict
from urllib.parse import urlparse
import logging

import dspy

from knowledge_storm.rm import YouRM, BingSearch, VectorRM
from .storm_dataclass import StormInformation
from ...interface import Retriever, Information
from ...utils import ArticleTextProcessing

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
    def __init__(self, rm, k: int = 3):
        super().__init__(search_top_k=k)
        self._rm = rm
        self.k = k
        logger.info(f"StormRetriever geïnitialiseerd met rm: {type(rm)} en k: {k}")
        if hasattr(rm, 'is_valid_source'):
            rm.is_valid_source = is_valid_wikipedia_source
            logger.info("is_valid_source methode overschreven voor rm")

    def retrieve(self, query: Union[str, List[str]], exclude_urls: List[str] = []) -> List[Union[StormInformation, Dict]]:
        logger.info(f"StormRetriever.retrieve aangeroepen met query: {query}, exclude_urls: {exclude_urls}")
        
        retrieved_data_list = []
        try:
            if isinstance(self._rm, CombinedRetriever):
                logger.debug("Gebruik CombinedRetriever.retrieve")
                retrieved_data_list = self._rm.retrieve(query, exclude_urls=exclude_urls)
            elif hasattr(self._rm, 'forward'):
                logger.debug("Gebruik rm.forward")
                retrieved_data_list = self._rm.forward(query, exclude_urls=exclude_urls)
            elif callable(self._rm):
                logger.debug("Gebruik rm direct als callable")
                retrieved_data_list = self._rm(query)
            else:
                logger.error(f"Onbekend type rm: {type(self._rm)}. Kan retrieve niet uitvoeren.")
                return []
        except Exception as e:
            logger.error(f"Fout bij ophalen van gegevens: {str(e)}", exc_info=True)
            return []
        
        logger.info(f"Aantal ontvangen resultaten: {len(retrieved_data_list)}")
        
        processed_results = []
        for i, data in enumerate(retrieved_data_list):
            logger.debug(f"Verwerken van resultaat {i+1}/{len(retrieved_data_list)}: {type(data)}")
            try:
                if isinstance(data, dict):
                    if 'snippets' in data:
                        logger.debug("Verwerken van You.com of Bing resultaat")
                        for j in range(len(data['snippets'])):
                            data['snippets'][j] = ArticleTextProcessing.remove_citations(data['snippets'][j])
                        processed_results.append(StormInformation.from_dict(data))
                    elif 'query' in data and 'answer' in data:
                        logger.debug("Verwerken van VectorRM resultaat")
                        processed_results.append(StormInformation.from_dict(data))
                    else:
                        logger.warning(f"Onbekend resultaatformaat: {data.keys()}. Dit resultaat wordt overgeslagen.")
                elif isinstance(data, StormInformation):
                    logger.debug("Resultaat is al een StormInformation object")
                    processed_results.append(data)
                else:
                    logger.warning(f"Onverwacht datatype: {type(data)}. Dit resultaat wordt overgeslagen.")
            except Exception as e:
                logger.error(f"Fout bij verwerken van resultaat {i+1}: {str(e)}", exc_info=True)
        
        logger.info(f"Aantal verwerkte resultaten: {len(processed_results)}")
        return processed_results[:self.k]

    def __call__(self, query: Union[str, List[str]], exclude_urls: List[str] = []) -> List[Union[StormInformation, Dict]]:
        logger.info("StormRetriever.__call__ aangeroepen, doorsturen naar retrieve methode")
        return self.retrieve(query, exclude_urls)

class CombinedRetriever(Retriever):
    def __init__(self, retrievers=None, k=3):
        super().__init__(search_top_k=k)
        self.k = k
        self.retrievers = retrievers or {}
        self.active_retrievers = set(self.retrievers.keys())
        logger.info(f"CombinedRetriever geïnitialiseerd met retrievers: {list(self.retrievers.keys())} en k: {k}")

    def set_active_retrievers(self, retriever_names):
        self.active_retrievers = set(retriever_names) & set(self.retrievers.keys())
        logger.info(f"Actieve retrievers ingesteld op: {self.active_retrievers}")

    def retrieve(self, query: Union[str, List[str]], exclude_urls: List[str] = []) -> List[Dict]:
        logger.info(f"CombinedRetriever.retrieve aangeroepen met query: {query}, exclude_urls: {exclude_urls}")
        
        combined_results = []
        for name in self.active_retrievers:
            retriever = self.retrievers[name]
            logger.debug(f"Ophalen resultaten van retriever: {name}")
            try:
                results = retriever.forward(query, exclude_urls=exclude_urls)
                logger.debug(f"Aantal resultaten van {name}: {len(results)}")
                
                for result in results:
                    if name == 'vector':
                        logger.debug(f"Verwerken van VectorRM resultaat van {name}")
                        combined_results.append({
                            'snippets': [result['answer']],
                            'score': 1.0,  # Standaardscore voor VectorRM resultaten
                            'url': '',  # Lege URL voor VectorRM resultaten
                            'title': f"VectorRM Antwoord: {result['query'][:30]}..."  # Titel gebaseerd op query
                        })
                    elif 'webPages' in result:
                        logger.debug(f"Verwerken van You.com of Bing resultaat van {name}")
                        for page in result['webPages']['value']:
                            combined_results.append({
                                'url': page['url'],
                                'title': page['name'],
                                'snippets': [page['snippet']]
                            })
                    else:
                        logger.debug(f"Onbekend resultaatformaat van {name}, toevoegen zonder aanpassingen")
                        combined_results.append(result)

            except Exception as e:
                logger.error(f"Fout bij ophalen resultaten van {name}: {str(e)}", exc_info=True)
        
        logger.info(f"Totaal aantal resultaten voor deduplicatie: {len(combined_results)}")
        
        final_results = combined_results[:self.k]
        logger.info(f"Aantal resultaten na beperking: {len(final_results)}")
        return final_results

    def update_search_top_k(self, k: int):
        self.k = k
        logger.info(f"search_top_k bijgewerkt naar {k}")
        for name, retriever in self.retrievers.items():
            if hasattr(retriever, 'update_search_top_k'):
                retriever.update_search_top_k(k)
                logger.debug(f"search_top_k bijgewerkt voor retriever {name}")

    def get_usage_and_reset(self):
        usage = {}
        for name, retriever in self.retrievers.items():
            if hasattr(retriever, 'get_usage_and_reset'):
                usage[name] = retriever.get_usage_and_reset()
        logger.debug(f"Gebruik opgehaald en gereset: {usage}")
        return usage