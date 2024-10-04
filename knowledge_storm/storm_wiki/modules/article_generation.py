import concurrent.futures
import copy
import logging
from concurrent.futures import as_completed
from typing import List, Union

import dspy

from .callback import BaseCallbackHandler
from .storm_dataclass import StormInformationTable, StormArticle
from ...interface import ArticleGenerationModule, Information
from ...utils import ArticleTextProcessing


class StormArticleGenerationModule(ArticleGenerationModule):
    """
    De interface voor de artikelgeneratiefase. Gegeven een onderwerp, verzamelde informatie uit
    de kenniscuratiefase en gegenereerde outline uit de outlinegeneratiefase.
    """

    def __init__(
        self,
        article_gen_lm=Union[dspy.dsp.LM, dspy.dsp.HFModel],
        retrieve_top_k: int = 5,
        max_thread_num: int = 10,
    ):
        super().__init__()
        self.retrieve_top_k = retrieve_top_k
        self.article_gen_lm = article_gen_lm
        self.max_thread_num = max_thread_num
        self.section_gen = ConvToSection(engine=self.article_gen_lm)

    def generate_section(
        self, topic, section_name, information_table, section_outline, section_query
    ):
        # Verzamel relevante informatie voor de sectie
        collected_info: List[Information] = []
        if information_table is not None:
            collected_info = information_table.retrieve_information(
                queries=section_query, search_top_k=self.retrieve_top_k
            )
        
        # Genereer de sectie-inhoud
        output = self.section_gen(
            topic=topic,
            outline=section_outline,
            section=section_name,
            collected_info=collected_info,
        )
        return {
            "section_name": section_name,
            "section_content": output.section,
            "collected_info": collected_info,
        }

    def generate_article(
        self,
        topic: str,
        information_table: StormInformationTable,
        article_with_outline: StormArticle,
        callback_handler: BaseCallbackHandler = None,
    ) -> StormArticle:
        """
        Genereer een artikel voor het onderwerp op basis van de informatietabel en artikeloutline.

        Args:
            topic (str): Het onderwerp van het artikel.
            information_table (StormInformationTable): De informatietabel met de verzamelde informatie.
            article_with_outline (StormArticle): Het artikel met gespecificeerde outline.
            callback_handler (BaseCallbackHandler): Een optionele callback handler die kan worden gebruikt om
                aangepaste callbacks te triggeren in verschillende stadia van het artikelgeneratieproces. Standaard None.
        """
        information_table.prepare_table_for_retrieval()

        if article_with_outline is None:
            article_with_outline = StormArticle(topic_name=topic)

        sections_to_write = article_with_outline.get_first_level_section_names()

        section_output_dict_collection = []
        if len(sections_to_write) == 0:
            logging.error(
                f"Geen outline voor {topic}. Zal direct zoeken met het onderwerp."
            )
            section_output_dict = self.generate_section(
                topic=topic,
                section_name=topic,
                information_table=information_table,
                section_outline="",
                section_query=[topic],
            )
            section_output_dict_collection = [section_output_dict]
        else:
            # Gebruik multithreading om secties parallel te genereren
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=self.max_thread_num
            ) as executor:
                future_to_sec_title = {}
                for section_title in sections_to_write:
                    # We willen geen aparte introductiesectie schrijven.
                    if section_title.lower().strip() == "introduction":
                        continue
                    # We willen geen aparte conclusiesectie schrijven.
                    if section_title.lower().strip().startswith(
                        "conclusion"
                    ) or section_title.lower().strip().startswith("summary"):
                        continue
                    
                    section_query = article_with_outline.get_outline_as_list(
                        root_section_name=section_title, add_hashtags=False
                    )
                    queries_with_hashtags = article_with_outline.get_outline_as_list(
                        root_section_name=section_title, add_hashtags=True
                    )
                    section_outline = "\n".join(queries_with_hashtags)
                    future_to_sec_title[
                        executor.submit(
                            self.generate_section,
                            topic,
                            section_title,
                            information_table,
                            section_outline,
                            section_query,
                        )
                    ] = section_title

                for future in as_completed(future_to_sec_title):
                    section_output_dict_collection.append(future.result())

        # Voeg gegenereerde secties toe aan het artikel
        article = copy.deepcopy(article_with_outline)
        for section_output_dict in section_output_dict_collection:
            article.update_section(
                parent_section_name=topic,
                current_section_content=section_output_dict["section_content"],
                current_section_info_list=section_output_dict["collected_info"],
            )
        article.post_processing()
        return article


class ConvToSection(dspy.Module):
    """Gebruik de informatie verzameld uit het informatiezoekgesprek om een sectie te schrijven."""

    def __init__(self, engine: Union[dspy.dsp.LM, dspy.dsp.HFModel]):
        super().__init__()
        self.write_section = dspy.Predict(WriteSection)
        self.engine = engine

    def forward(
        self, topic: str, outline: str, section: str, collected_info: List[Information]
    ):
        # Bereid verzamelde informatie voor
        info = ""
        for idx, storm_info in enumerate(collected_info):
            info += f"[{idx + 1}]\n" + "\n".join(storm_info.snippets)
            info += "\n\n"

        # Beperk de hoeveelheid informatie om overbelasting te voorkomen
        info = ArticleTextProcessing.limit_word_count_preserve_newline(info, 1500)

        # Genereer de sectie-inhoud
        with dspy.settings.context(lm=self.engine):
            section = ArticleTextProcessing.clean_up_section(
                self.write_section(topic=topic, info=info, section=section).output
            )

        return dspy.Prediction(section=section)


class WriteSection(dspy.Signature):
    """
    Schrijf een Wikipedia-sectie op basis van de verzamelde informatie.

    Hier is het formaat van je schrijven:
        1. Gebruik "#" Titel" om een sectietitel aan te geven, "##" Titel" voor een subsectietitel, "###" Titel" voor een subsubsectietitel, enzovoort.
        2. Gebruik [1], [2], ..., [n] in de regel (bijvoorbeeld, "De hoofdstad van de Verenigde Staten is Washington, D.C.[1][3]."). Je hoeft GEEN Referenties- of Bronnensectie op te nemen om de bronnen aan het einde te vermelden.
    """

    info = dspy.InputField(prefix="De verzamelde informatie:\n", format=str)
    topic = dspy.InputField(prefix="Het onderwerp van de pagina: ", format=str)
    section = dspy.InputField(prefix="De sectie die je moet schrijven: ", format=str)
    output = dspy.OutputField(
        prefix="Schrijf de sectie met juiste inline citaties (Begin je schrijven met # sectietitel. Neem de paginatitel niet op en probeer geen andere secties te schrijven):\n",
        format=str,
    )