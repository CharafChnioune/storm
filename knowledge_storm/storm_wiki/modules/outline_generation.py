from typing import Union, Optional, Tuple

import dspy

from .callback import BaseCallbackHandler
from .storm_dataclass import StormInformationTable, StormArticle
from ...interface import OutlineGenerationModule
from ...utils import ArticleTextProcessing


class StormOutlineGenerationModule(OutlineGenerationModule):
    """
    De interface voor de outline generatie fase. Gegeven een onderwerp en verzamelde informatie
    uit de kenniscuratie fase, genereert deze module een outline voor het artikel.
    """

    def __init__(self, outline_gen_lm: Union[dspy.dsp.LM, dspy.dsp.HFModel]):
        super().__init__()
        self.outline_gen_lm = outline_gen_lm
        self.write_outline = WriteOutline(engine=self.outline_gen_lm)

    def generate_outline(
        self,
        topic: str,
        information_table: StormInformationTable,
        old_outline: Optional[StormArticle] = None,
        callback_handler: BaseCallbackHandler = None,
        return_draft_outline=False,
    ) -> Union[StormArticle, Tuple[StormArticle, StormArticle]]:
        """
        Genereert een outline voor een artikel op basis van het opgegeven onderwerp en de
        verzamelde informatie tijdens de kenniscuratie fase. Deze methode kan optioneel zowel
        de uiteindelijke artikel-outline als een concept-outline retourneren indien gewenst.

        Args:
            topic (str): Het onderwerp van het artikel.
            information_table (StormInformationTable): De informatietabel met de verzamelde gegevens.
            old_outline (Optional[StormArticle]): Een optionele vorige versie van de artikel-outline
                die kan worden gebruikt als referentie of vergelijking. Standaard is None.
            callback_handler (BaseCallbackHandler): Een optionele callback handler die kan worden
                gebruikt om aangepaste callbacks te activeren op verschillende momenten in het
                outline generatieproces, zoals wanneer de informatie-organisatie begint. Standaard is None.
            return_draft_outline (bool): Een vlag die aangeeft of de methode zowel de uiteindelijke
                artikel-outline als een conceptversie van de outline moet retourneren. Als False,
                wordt alleen de uiteindelijke artikel-outline geretourneerd. Standaard is False.

        Returns:
            Union[StormArticle, Tuple[StormArticle, StormArticle]]: Afhankelijk van de waarde van
            `return_draft_outline`, retourneert deze methode ofwel een enkel `StormArticle` object
            met de uiteindelijke outline, ofwel een tuple van twee `StormArticle` objecten, waarbij
            de eerste de uiteindelijke outline bevat en de tweede de concept-outline.
        """
        if callback_handler is not None:
            callback_handler.on_information_organization_start()

        # Combineer alle conversatiebeurten tot één lijst
        concatenated_dialogue_turns = sum(
            [conv for (_, conv) in information_table.conversations], []
        )
        
        # Genereer de outline met behulp van de WriteOutline module
        result = self.write_outline(
            topic=topic,
            dlg_history=concatenated_dialogue_turns,
            callback_handler=callback_handler,
        )
        
        # Creëer StormArticle objecten voor de uiteindelijke en concept-outline
        article_with_outline_only = StormArticle.from_outline_str(
            topic=topic, outline_str=result.outline
        )
        article_with_draft_outline_only = StormArticle.from_outline_str(
            topic=topic, outline_str=result.old_outline
        )
        
        # Retourneer de gewenste output op basis van return_draft_outline
        if not return_draft_outline:
            return article_with_outline_only
        return article_with_outline_only, article_with_draft_outline_only


class WriteOutline(dspy.Module):
    """Genereer de outline voor de Wikipedia-pagina."""

    def __init__(self, engine: Union[dspy.dsp.LM, dspy.dsp.HFModel]):
        super().__init__()
        self.draft_page_outline = dspy.Predict(WritePageOutline)
        self.write_page_outline = dspy.Predict(WritePageOutlineFromConv)
        self.engine = engine

    def forward(
        self,
        topic: str,
        dlg_history,
        old_outline: Optional[str] = None,
        callback_handler: BaseCallbackHandler = None,
    ):
        # Verwijder irrelevante dialoogbeurten
        trimmed_dlg_history = []
        for turn in dlg_history:
            if (
                "topic you" in turn.agent_utterance.lower()
                or "topic you" in turn.user_utterance.lower()
            ):
                continue
            trimmed_dlg_history.append(turn)
        
        # Converteer de dialooggeschiedenis naar een string
        conv = "\n".join(
            [
                f"Wikipedia Writer: {turn.user_utterance}\nExpert: {turn.agent_utterance}"
                for turn in trimmed_dlg_history
            ]
        )
        
        # Verwijder citaten en beperk het aantal woorden
        conv = ArticleTextProcessing.remove_citations(conv)
        conv = ArticleTextProcessing.limit_word_count_preserve_newline(conv, 5000)

        with dspy.settings.context(lm=self.engine):
            # Genereer een concept-outline als er geen oude outline is
            if old_outline is None:
                old_outline = ArticleTextProcessing.clean_up_outline(
                    self.draft_page_outline(topic=topic).outline
                )
                if callback_handler:
                    callback_handler.on_direct_outline_generation_end(
                        outline=old_outline
                    )
            
            # Genereer de uiteindelijke outline op basis van de conversatie en de oude outline
            outline = ArticleTextProcessing.clean_up_outline(
                self.write_page_outline(
                    topic=topic, old_outline=old_outline, conv=conv
                ).outline
            )
            if callback_handler:
                callback_handler.on_outline_refinement_end(outline=outline)

        return dspy.Prediction(outline=outline, old_outline=old_outline)


class WritePageOutline(dspy.Signature):
    """
    Schrijf een outline voor een Wikipedia-pagina.
    Hier is het formaat van je schrijven:
    1. Gebruik "#" Titel" om een sectietitel aan te geven, "##" Titel" voor een subsectietitel, "###" Titel" voor een subsubsectietitel, enzovoort.
    2. Voeg geen andere informatie toe.
    3. Neem de onderwerpnaam zelf niet op in de outline.
    """

    topic = dspy.InputField(prefix="Het onderwerp waarover je wilt schrijven: ", format=str)
    outline = dspy.OutputField(prefix="Schrijf de Wikipedia-pagina outline:\n", format=str)


class NaiveOutlineGen(dspy.Module):
    """Genereer de outline direct met de parametrische kennis van het LLM."""

    def __init__(self):
        super().__init__()
        self.write_outline = dspy.Predict(WritePageOutline)

    def forward(self, topic: str):
        outline = self.write_outline(topic=topic).outline

        return dspy.Prediction(outline=outline)


class WritePageOutlineFromConv(dspy.Signature):
    """
    Verbeter een outline voor een Wikipedia-pagina. Je hebt al een concept-outline die de algemene informatie behandelt.
    Nu wil je deze verbeteren op basis van de informatie die je hebt geleerd uit een informatiezoekend gesprek om het informatiever te maken.
    Hier is het formaat van je schrijven:
    1. Gebruik "#" Titel" om een sectietitel aan te geven, "##" Titel" voor een subsectietitel, "###" Titel" voor een subsubsectietitel, enzovoort.
    2. Voeg geen andere informatie toe.
    3. Neem de onderwerpnaam zelf niet op in de outline.
    """

    topic = dspy.InputField(prefix="Het onderwerp waarover je wilt schrijven: ", format=str)
    conv = dspy.InputField(prefix="Gespreksgeschiedenis:\n", format=str)
    old_outline = dspy.OutputField(prefix="Huidige outline:\n", format=str)
    outline = dspy.OutputField(
        prefix='Schrijf de Wikipedia-pagina outline (Gebruik "#" Titel" om een sectietitel aan te geven, "##" Titel" voor een subsectietitel, ...):\n',
        format=str,
    )