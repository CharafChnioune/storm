import logging
import os
from typing import Union, Optional, Tuple

import dspy

from .callback import BaseCallbackHandler
from .storm_dataclass import StormInformationTable, StormArticle
from ...interface import OutlineGenerationModule
from ...utils import ArticleTextProcessing

# Configureer logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class StormOutlineGenerationModule(OutlineGenerationModule):
    """
    De interface voor de outline generatie fase. Gegeven het onderwerp en verzamelde informatie uit de
    kenniscuratie fase, genereert een outline voor het artikel.
    """

    def __init__(self, outline_gen_lm: Union[dspy.dsp.LM, dspy.dsp.HFModel]):
        super().__init__()
        self.outline_gen_lm = outline_gen_lm
        self.write_outline = WriteOutline(engine=self.outline_gen_lm)
        logger.info("StormOutlineGenerationModule geïnitialiseerd")

    def generate_outline(self, topic: str, information_table: StormInformationTable,
                         old_outline: Optional[StormArticle] = None,
                         callback_handler: BaseCallbackHandler = None,
                         return_draft_outline=False) -> Union[StormArticle, Tuple[StormArticle, StormArticle]]:
        logger.info(f"Start outline generatie voor onderwerp: {topic}")
        if callback_handler is not None:
            callback_handler.on_information_organization_start()
            logger.info("Information organization gestart")

        concatenated_dialogue_turns = sum([conv for (_, conv) in information_table.conversations], [])
        logger.info(f"Aantal dialoogbeurten: {len(concatenated_dialogue_turns)}")

        result = self.write_outline(topic=topic, dlg_history=concatenated_dialogue_turns,
                                    callback_handler=callback_handler)
        logger.info("Outline generatie voltooid")

        article_with_outline_only = StormArticle.from_outline_str(topic=topic, outline_str=result.outline)
        article_with_draft_outline_only = StormArticle.from_outline_str(topic=topic,
                                                                        outline_str=result.old_outline)
        logger.info("Artikelen met outlines gemaakt")

        if not return_draft_outline:
            logger.info("Retourneren van definitieve outline")
            return article_with_outline_only
        logger.info("Retourneren van definitieve en concept outline")
        return article_with_outline_only, article_with_draft_outline_only

class WriteOutline(dspy.Module):
    """Genereer de outline voor de Wikipedia pagina."""

    def __init__(self, engine: Union[dspy.dsp.LM, dspy.dsp.HFModel]):
        super().__init__()
        self.draft_page_outline = dspy.Predict(WritePageOutline)
        self.write_page_outline = dspy.Predict(WritePageOutlineFromConv)
        self.engine = engine
        logger.info("WriteOutline module geïnitialiseerd")

    def forward(self, topic: str, dlg_history, old_outline: Optional[str] = None,
                callback_handler: BaseCallbackHandler = None):
        logger.info(f"Start outline schrijven voor onderwerp: {topic}")
        trimmed_dlg_history = []
        for turn in dlg_history:
            if 'topic you' in turn.agent_utterance.lower() or 'topic you' in turn.user_utterance.lower():
                continue
            trimmed_dlg_history.append(turn)
        logger.info(f"Aantal getrimde dialoogbeurten: {len(trimmed_dlg_history)}")

        conv = '\n'.join([f'Wikipedia Writer: {turn.user_utterance}\nExpert: {turn.agent_utterance}' for turn in
                          trimmed_dlg_history])
        conv = ArticleTextProcessing.remove_citations(conv)
        conv = ArticleTextProcessing.limit_word_count_preserve_newline(conv, 5000)
        logger.info("Conversatie voorbereid voor outline generatie")

        with dspy.settings.context(lm=self.engine):
            if old_outline is None:
                logger.info("Genereren van concept outline")
                old_outline = ArticleTextProcessing.clean_up_outline(self.draft_page_outline(topic=topic).outline)
                if callback_handler:
                    callback_handler.on_direct_outline_generation_end(outline=old_outline)
                    logger.info("Concept outline generatie voltooid")
            logger.info("Genereren van definitieve outline")
            outline = ArticleTextProcessing.clean_up_outline(
                self.write_page_outline(topic=topic, old_outline=old_outline, conv=conv).outline)
            if callback_handler:
                callback_handler.on_outline_refinement_end(outline=outline)
                logger.info("Definitieve outline generatie voltooid")

        return dspy.Prediction(outline=outline, old_outline=old_outline)

class WritePageOutline(dspy.Signature):
    """Schrijf een outline voor een Wikipedia pagina."""

    topic = dspy.InputField(prefix="Het onderwerp waarover je wilt schrijven: ", format=str)
    outline = dspy.OutputField(prefix="Schrijf de Wikipedia pagina outline:\n", format=str)

class NaiveOutlineGen(dspy.Module):
    """Genereer de outline direct met LLM's parametrische kennis."""

    def __init__(self):
        super().__init__()
        self.write_outline = dspy.Predict(WritePageOutline)
        logger.info("NaiveOutlineGen module geïnitialiseerd")

    def forward(self, topic: str):
        logger.info(f"Start naïeve outline generatie voor onderwerp: {topic}")
        outline = self.write_outline(topic=topic).outline
        logger.info("Naïeve outline generatie voltooid")
        return dspy.Prediction(outline=outline)

class WritePageOutlineFromConv(dspy.Signature):
    """Verbeter een outline voor een Wikipedia pagina."""

    topic = dspy.InputField(prefix="Het onderwerp waarover je wilt schrijven: ", format=str)
    conv = dspy.InputField(prefix="Conversatiegeschiedenis:\n", format=str)
    old_outline = dspy.OutputField(prefix="Huidige outline:\n", format=str)
    outline = dspy.OutputField(
        prefix='Schrijf de Wikipedia pagina outline (Gebruik "#" Titel" voor sectietitel, "##" Titel" voor subsectietitel, ...):\n',
        format=str
    )

# Voeg een handler toe om logs naar een bestand te schrijven
file_handler = logging.FileHandler('storm_outline_generation.log')
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logger.addHandler(file_handler)

logger.info("Script geïnitialiseerd en logging geconfigureerd")