import copy
import logging
from typing import Union

import dspy

from .storm_dataclass import StormArticle
from ...interface import ArticlePolishingModule
from ...utils import ArticleTextProcessing

# Configureer logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class StormArticlePolishingModule(ArticlePolishingModule):
    """
    De interface voor de artikel generatie fase. Gegeven onderwerp, verzamelde informatie uit
    kenniscuratie fase, gegenereerde outline uit outline generatie fase.
    """

    def __init__(self,
                 article_gen_lm: Union[dspy.dsp.LM, dspy.dsp.HFModel],
                 article_polish_lm: Union[dspy.dsp.LM, dspy.dsp.HFModel]):
        logger.info("Initializing StormArticlePolishingModule")
        self.article_gen_lm = article_gen_lm
        self.article_polish_lm = article_polish_lm

        self.polish_page = PolishPageModule(
            write_lead_engine=self.article_gen_lm,
            polish_engine=self.article_polish_lm
        )
        logger.debug("StormArticlePolishingModule initialized successfully")

    def polish_article(self,
                       topic: str,
                       draft_article: StormArticle,
                       remove_duplicate: bool = False) -> StormArticle:
        """
        Polish artikel.

        Args:
            topic (str): Het onderwerp van het artikel.
            draft_article (StormArticle): Het concept artikel.
            remove_duplicate (bool): Of een extra LM-aanroep moet worden gebruikt om duplicaten uit het artikel te verwijderen.
        """
        logger.info(f"Starting article polishing for topic: {topic}")
        logger.debug(f"Remove duplicate setting: {remove_duplicate}")

        article_text = draft_article.to_string()
        logger.debug(f"Draft article length: {len(article_text)} characters")

        polish_result = self.polish_page(topic=topic, draft_page=article_text, polish_whole_page=remove_duplicate)
        logger.info("Article polishing completed")

        lead_section = f"# summary\n{polish_result.lead_section}"
        logger.debug(f"Lead section length: {len(lead_section)} characters")

        polished_article = '\n\n'.join([lead_section, polish_result.page])
        logger.debug(f"Polished article length: {len(polished_article)} characters")

        polished_article_dict = ArticleTextProcessing.parse_article_into_dict(polished_article)
        logger.debug(f"Parsed article into dictionary with {len(polished_article_dict)} top-level sections")

        polished_article = copy.deepcopy(draft_article)
        polished_article.insert_or_create_section(article_dict=polished_article_dict)
        logger.info("Inserted polished content into article structure")

        polished_article.post_processing()
        logger.info("Post-processing of polished article completed")

        return polished_article


class WriteLeadSection(dspy.Signature):
    """Schrijf een lead sectie voor de gegeven Wikipedia pagina met de volgende richtlijnen:
        1. De lead moet op zichzelf staan als een beknopt overzicht van het onderwerp van het artikel. Het moet het onderwerp identificeren, context geven, uitleggen waarom het onderwerp opmerkelijk is, en de belangrijkste punten samenvatten, inclusief eventuele prominente controverses.
        2. De lead sectie moet beknopt zijn en niet meer dan vier goed samengestelde alinea's bevatten.
        3. De lead sectie moet zorgvuldig worden onderbouwd waar nodig. Voeg inline citaties toe (bijv. "Washington, D.C., is de hoofdstad van de Verenigde Staten.[1][3].") waar nodig."""

    topic = dspy.InputField(prefix="Het onderwerp van de pagina: ", format=str)
    draft_page = dspy.InputField(prefix="De concept pagina:\n", format=str)
    lead_section = dspy.OutputField(prefix="Schrijf de lead sectie:\n", format=str)


class PolishPage(dspy.Signature):
    """Je bent een getrouwe tekstredacteur die goed is in het vinden van herhaalde informatie in het artikel en deze te verwijderen om ervoor te zorgen dat er geen herhaling in het artikel voorkomt. Je zult geen enkel niet-herhaald deel in het artikel verwijderen. Je zult de inline citaties en artikelstructuur (aangegeven door "#", "##", etc.) op gepaste wijze behouden. Doe je werk voor het volgende artikel."""

    draft_page = dspy.InputField(prefix="Het concept artikel:\n", format=str)
    page = dspy.OutputField(prefix="Je herziene artikel:\n", format=str)


class PolishPageModule(dspy.Module):
    def __init__(self, write_lead_engine: Union[dspy.dsp.LM, dspy.dsp.HFModel],
                 polish_engine: Union[dspy.dsp.LM, dspy.dsp.HFModel]):
        super().__init__()
        logger.info("Initializing PolishPageModule")
        self.write_lead_engine = write_lead_engine
        self.polish_engine = polish_engine
        self.write_lead = dspy.Predict(WriteLeadSection)
        self.polish_page = dspy.Predict(PolishPage)
        logger.debug("PolishPageModule initialized successfully")

    def forward(self, topic: str, draft_page: str, polish_whole_page: bool = True):
        logger.info(f"Starting forward pass for topic: {topic}")
        logger.debug(f"Polish whole page setting: {polish_whole_page}")

        with dspy.settings.context(lm=self.write_lead_engine):
            logger.debug("Generating lead section")
            lead_section = self.write_lead(topic=topic, draft_page=draft_page).lead_section
            if "The lead section:" in lead_section:
                lead_section = lead_section.split("The lead section:")[1].strip()
            logger.info("Lead section generation completed")
            logger.debug(f"Lead section length: {len(lead_section)} characters")

        if polish_whole_page:
            logger.info("Polishing whole page")
            with dspy.settings.context(lm=self.polish_engine):
                page = self.polish_page(draft_page=draft_page).page
            logger.debug(f"Polished page length: {len(page)} characters")
        else:
            logger.info("Skipping whole page polishing")
            page = draft_page

        logger.info("Forward pass completed")
        return dspy.Prediction(lead_section=lead_section, page=page)

# Voeg een handler toe om logs naar een bestand te schrijven
file_handler = logging.FileHandler('storm_article_polishing.log')
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
logger.addHandler(file_handler)

logger.info("StormArticlePolishingModule script initialized and logging configured")