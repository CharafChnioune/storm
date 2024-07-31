import concurrent.futures
import copy
import logging
from concurrent.futures import as_completed
from typing import List, Union

import dspy

from .callback import BaseCallbackHandler
from .storm_dataclass import StormInformationTable, StormArticle, StormInformation
from ...interface import ArticleGenerationModule
from ...utils import ArticleTextProcessing

# Configureer logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class StormArticleGenerationModule(ArticleGenerationModule):
    """
    De interface voor de artikel generatie fase. Gegeven onderwerp, verzamelde informatie uit
    kenniscuratie fase, gegenereerde outline uit outline generatie fase.
    """

    def __init__(self,
                 article_gen_lm=Union[dspy.dsp.LM, dspy.dsp.HFModel],
                 retrieve_top_k: int = 5,
                 max_thread_num: int = 10):
        super().__init__()
        self.retrieve_top_k = retrieve_top_k
        self.article_gen_lm = article_gen_lm
        self.max_thread_num = max_thread_num
        self.section_gen = ConvToSection(engine=self.article_gen_lm)
        logger.info(f"StormArticleGenerationModule initialized with retrieve_top_k={retrieve_top_k}, max_thread_num={max_thread_num}")

    def generate_section(self, topic, section_name, information_table, section_outline, section_query):
        logger.info(f"Generating section: {section_name} for topic: {topic}")
        collected_info: List[StormInformation] = []
        if information_table is not None:
            collected_info = information_table.retrieve_information(queries=section_query,
                                                                    search_top_k=self.retrieve_top_k)
            logger.debug(f"Retrieved {len(collected_info)} pieces of information for section: {section_name}")
        
        output = self.section_gen(topic=topic,
                                  outline=section_outline,
                                  section=section_name,
                                  collected_info=collected_info)
        logger.info(f"Section generation completed for: {section_name}")
        return {"section_name": section_name, "section_content": output.section, "collected_info": collected_info}

    def generate_article(self,
                         topic: str,
                         information_table: StormInformationTable,
                         article_with_outline: StormArticle,
                         callback_handler: BaseCallbackHandler = None) -> StormArticle:
        """
        Genereer artikel voor het onderwerp op basis van de informatietabel en artikel outline.
        """
        logger.info(f"Starting article generation for topic: {topic}")
        information_table.prepare_table_for_retrieval()
        logger.debug("Information table prepared for retrieval")

        if article_with_outline is None:
            article_with_outline = StormArticle(topic_name=topic)
            logger.warning(f"No outline provided for {topic}. Created a new StormArticle.")

        sections_to_write = article_with_outline.get_first_level_section_names()
        logger.info(f"Number of sections to write: {len(sections_to_write)}")

        section_output_dict_collection = []
        if len(sections_to_write) == 0:
            logger.error(f'No outline for {topic}. Will directly search with the topic.')
            section_output_dict = self.generate_section(
                topic=topic,
                section_name=topic,
                information_table=information_table,
                section_outline="",
                section_query=[topic]
            )
            section_output_dict_collection = [section_output_dict]
        else:
            logger.info("Starting concurrent section generation")
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_thread_num) as executor:
                future_to_sec_title = {}
                for section_title in sections_to_write:
                    if section_title.lower().strip() in ['introduction', 'conclusion', 'summary']:
                        logger.debug(f"Skipping section: {section_title}")
                        continue
                    
                    section_query = article_with_outline.get_outline_as_list(root_section_name=section_title,
                                                                             add_hashtags=False)
                    queries_with_hashtags = article_with_outline.get_outline_as_list(
                        root_section_name=section_title, add_hashtags=True)
                    section_outline = "\n".join(queries_with_hashtags)
                    
                    future = executor.submit(self.generate_section,
                                             topic, section_title, information_table, section_outline, section_query)
                    future_to_sec_title[future] = section_title
                    logger.debug(f"Submitted section generation task for: {section_title}")

                for future in as_completed(future_to_sec_title):
                    section_title = future_to_sec_title[future]
                    try:
                        section_output_dict_collection.append(future.result())
                        logger.info(f"Section generation completed for: {section_title}")
                    except Exception as exc:
                        logger.error(f"Section generation failed for {section_title}: {exc}")

        logger.info("Section generation completed. Starting article assembly.")
        article = copy.deepcopy(article_with_outline)
        for section_output_dict in section_output_dict_collection:
            article.update_section(parent_section_name=topic,
                                   current_section_content=section_output_dict["section_content"],
                                   current_section_info_list=section_output_dict["collected_info"])
            logger.debug(f"Updated section: {section_output_dict['section_name']}")
        
        article.post_processing()
        logger.info("Article post-processing completed")
        return article


class ConvToSection(dspy.Module):
    """Gebruik de informatie verzameld uit het informatiezoekgesprek om een sectie te schrijven."""

    def __init__(self, engine: Union[dspy.dsp.LM, dspy.dsp.HFModel]):
        super().__init__()
        self.write_section = dspy.Predict(WriteSection)
        self.engine = engine
        logger.info("ConvToSection module initialized")

    def forward(self, topic: str, outline: str, section: str, collected_info: List[StormInformation]):
        logger.info(f"Starting section generation for: {section}")
        info = ''
        for idx, storm_info in enumerate(collected_info):
            info += f'[{idx + 1}]\n' + '\n'.join(storm_info.snippets)
            info += '\n\n'

        info = ArticleTextProcessing.limit_word_count_preserve_newline(info, 1500)
        logger.debug(f"Prepared information for section {section}. Length: {len(info)} characters")

        with dspy.settings.context(lm=self.engine):
            section_content = self.write_section(topic=topic, info=info, section=section).output
            section_content = ArticleTextProcessing.clean_up_section(section_content)
            logger.info(f"Section generation completed for: {section}")
            logger.debug(f"Generated section length: {len(section_content)} characters")

        return dspy.Prediction(section=section_content)


class WriteSection(dspy.Signature):
    """Schrijf een Wikipedia-sectie op basis van de verzamelde informatie."""

    info = dspy.InputField(prefix="De verzamelde informatie:\n", format=str)
    topic = dspy.InputField(prefix="Het onderwerp van de pagina: ", format=str)
    section = dspy.InputField(prefix="De sectie die je moet schrijven: ", format=str)
    output = dspy.OutputField(
        prefix="Schrijf de sectie met juiste inline citaties (Begin je schrijven met # sectietitel. Voeg niet de paginatitel toe of probeer geen andere secties te schrijven):\n",
        format=str
    )

# Voeg een handler toe om logs naar een bestand te schrijven
file_handler = logging.FileHandler('storm_article_generation.log')
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
logger.addHandler(file_handler)

logger.info("StormArticleGenerationModule script initialized and logging configured")