import copy
from typing import Union

import dspy

from .storm_dataclass import StormArticle
from ...interface import ArticlePolishingModule
from ...utils import ArticleTextProcessing


class StormArticlePolishingModule(ArticlePolishingModule):
    """
    De interface voor de artikelpolijstfase. Gegeven een onderwerp, verzamelde informatie uit
    de kenniscuratiefase en gegenereerde outline uit de outlinegeneratiefase, polijst deze module het artikel.
    """

    def __init__(
        self,
        article_gen_lm: Union[dspy.dsp.LM, dspy.dsp.HFModel],
        article_polish_lm: Union[dspy.dsp.LM, dspy.dsp.HFModel],
    ):
        # Initialiseer taalmodellen voor artikelgeneratie en -polijsting
        self.article_gen_lm = article_gen_lm
        self.article_polish_lm = article_polish_lm

        # Maak een instantie van PolishPageModule met de gespecificeerde taalmodellen
        self.polish_page = PolishPageModule(
            write_lead_engine=self.article_gen_lm, polish_engine=self.article_polish_lm
        )

    def polish_article(
        self, topic: str, draft_article: StormArticle, remove_duplicate: bool = False
    ) -> StormArticle:
        """
        Polijst een artikel.

        Args:
            topic (str): Het onderwerp van het artikel.
            draft_article (StormArticle): Het conceptartikel.
            remove_duplicate (bool): Of een extra LM-aanroep moet worden gebruikt om duplicaten uit het artikel te verwijderen.

        Returns:
            StormArticle: Het gepolijste artikel.
        """

        # Converteer het conceptartikel naar een string
        article_text = draft_article.to_string()

        # Polijst de pagina met de PolishPageModule
        polish_result = self.polish_page(
            topic=topic, draft_page=article_text, polish_whole_page=remove_duplicate
        )

        # Voeg de inleidende sectie toe aan het gepolijste artikel
        lead_section = f"# summary\n{polish_result.lead_section}"
        polished_article = "\n\n".join([lead_section, polish_result.page])

        # Converteer het gepolijste artikel terug naar een StormArticle object
        polished_article_dict = ArticleTextProcessing.parse_article_into_dict(
            polished_article
        )
        polished_article = copy.deepcopy(draft_article)
        polished_article.insert_or_create_section(article_dict=polished_article_dict)
        polished_article.post_processing()

        return polished_article


class WriteLeadSection(dspy.Signature):
    """
    Schrijf een inleidende sectie voor de gegeven Wikipedia-pagina met de volgende richtlijnen:
    1. De inleiding moet op zichzelf staan als een beknopt overzicht van het onderwerp van het artikel. 
       Het moet het onderwerp identificeren, context bieden, uitleggen waarom het onderwerp opmerkelijk is, 
       en de belangrijkste punten samenvatten, inclusief eventuele prominente controverses.
    2. De inleidende sectie moet beknopt zijn en niet meer dan vier goed samengestelde alinea's bevatten.
    3. De inleidende sectie moet zorgvuldig worden onderbouwd waar nodig. Voeg inline citaties toe 
       (bijv. "Washington, D.C., is de hoofdstad van de Verenigde Staten.[1][3].") waar nodig.
    """

    topic = dspy.InputField(prefix="Het onderwerp van de pagina: ", format=str)
    draft_page = dspy.InputField(prefix="De conceptpagina:\n", format=str)
    lead_section = dspy.OutputField(prefix="Schrijf de inleidende sectie:\n", format=str)


class PolishPage(dspy.Signature):
    """
    Je bent een nauwkeurige tekstredacteur die goed is in het vinden van herhaalde informatie in het artikel 
    en het verwijderen ervan om ervoor te zorgen dat er geen herhaling in het artikel voorkomt. Je zult geen 
    niet-herhaalde delen in het artikel verwijderen. Je zult de inline citaties en artikelstructuur 
    (aangegeven door "#", "##", enz.) op de juiste manier behouden. Doe je werk voor het volgende artikel.
    """

    draft_page = dspy.InputField(prefix="Het conceptartikel:\n", format=str)
    page = dspy.OutputField(prefix="Je herziene artikel:\n", format=str)


class PolishPageModule(dspy.Module):
    def __init__(
        self,
        write_lead_engine: Union[dspy.dsp.LM, dspy.dsp.HFModel],
        polish_engine: Union[dspy.dsp.LM, dspy.dsp.HFModel],
    ):
        super().__init__()
        # Initialiseer taalmodellen voor het schrijven van de inleiding en het polijsten van de pagina
        self.write_lead_engine = write_lead_engine
        self.polish_engine = polish_engine
        
        # Maak voorspellingsmodules voor het schrijven van de inleiding en het polijsten van de pagina
        self.write_lead = dspy.Predict(WriteLeadSection)
        self.polish_page = dspy.Predict(PolishPage)

    def forward(self, topic: str, draft_page: str, polish_whole_page: bool = True):
        # Genereer de inleidende sectie
        # OPMERKING: Verander show_guidelines naar false om de generatie robuuster te maken voor verschillende LM-families
        with dspy.settings.context(lm=self.write_lead_engine, show_guidelines=False):
            lead_section = self.write_lead(
                topic=topic, draft_page=draft_page
            ).lead_section
            if "The lead section:" in lead_section:
                lead_section = lead_section.split("The lead section:")[1].strip()
        
        # Polijst de hele pagina indien nodig
        if polish_whole_page:
            # OPMERKING: Verander show_guidelines naar false om de generatie robuuster te maken voor verschillende LM-families
            with dspy.settings.context(lm=self.polish_engine, show_guidelines=False):
                page = self.polish_page(draft_page=draft_page).page
        else:
            page = draft_page

        return dspy.Prediction(lead_section=lead_section, page=page)