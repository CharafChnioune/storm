"""
Deze module behandelt vraag-generatie binnen het Co-STORM framework, specifiek ontworpen om de Moderator-rol te ondersteunen.

De Moderator genereert inzichtelijke, prikkelende vragen die nieuwe richtingen in het gesprek introduceren.
Door gebruik te maken van ongebruikte of niet-geciteerde informatiesnippers die tijdens de discussie zijn opgehaald, zorgt de Moderator ervoor dat het gesprek dynamisch blijft en herhalingen of te specifieke onderwerpen vermijdt.

Voor meer gedetailleerde informatie, zie Sectie 3.5 van het Co-STORM paper: https://www.arxiv.org/pdf/2408.15232.
"""

import dspy
from typing import List, Union

from .collaborative_storm_utils import (
    format_search_results,
    extract_and_remove_citations,
    keep_first_and_last_paragraph,
    extract_cited_storm_info,
)
from ...dataclass import ConversationTurn, KnowledgeBase
from ...interface import Information


class KnowledgeBaseSummmary(dspy.Signature):
    """Je taak is om een korte samenvatting te geven van wat er besproken is in een rondetafelgesprek. 
    De inhoud is thematisch georganiseerd in hiërarchische secties.
    Je krijgt deze secties te zien waarbij "#" het niveau van de sectie aangeeft.
    """

    topic = dspy.InputField(prefix="onderwerp: ", format=str)
    structure = dspy.InputField(prefix="Boomstructuur: \n", format=str)
    output = dspy.OutputField(prefix="Geef nu een korte samenvatting:\n", format=str)


class ConvertUtteranceStyle(dspy.Signature):
    """
    Je bent een uitgenodigd spreker in het rondetafelgesprek.
    Je taak is om de vraag of het antwoord meer conversationeel en boeiend te maken om de gespreksflow te bevorderen.
    Merk op dat dit een lopend gesprek is, dus geen behoefte aan welkomst- en afsluitende woorden. De uitspraak van de vorige spreker wordt alleen gegeven om het gesprek natuurlijker te maken.
    Let op dat je niet hallucineert en houd de citaatindex zoals [1] intact.
    """

    expert = dspy.InputField(prefix="Je bent uitgenodigd als: ", format=str)
    action = dspy.InputField(
        prefix="Je wilt bijdragen aan het gesprek door: ", format=str
    )
    prev = dspy.InputField(prefix="De vorige spreker zei: ", format=str)
    content = dspy.InputField(
        prefix="Vraag of antwoord dat je wilt zeggen: ", format=str
    )
    utterance = dspy.OutputField(
        prefix="Jouw uitspraak (behoud zoveel mogelijk informatie met citaten, geef de voorkeur aan kortere antwoorden zonder informatieverlies): ",
        format=str,
    )


class GroundedQuestionGeneration(dspy.Signature):
    """Je taak is om de volgende discussiefocus in een rondetafelgesprek te vinden. Je krijgt een samenvatting van het vorige gesprek en wat informatie die je kan helpen bij het ontdekken van een nieuwe discussiefocus.
    Let op dat de nieuwe discussiefocus een nieuwe invalshoek en perspectief in de discussie moet brengen en herhaling moet vermijden. De nieuwe discussiefocus moet gebaseerd zijn op de beschikbare informatie en de grenzen van de huidige discussie verleggen voor een bredere verkenning.
    De nieuwe discussiefocus moet een natuurlijke flow hebben vanaf de laatste uitspraak in het gesprek.
    Gebruik [1][2] in de tekst om je vraag te onderbouwen.
    """

    topic = dspy.InputField(prefix="onderwerp: ", format=str)
    summary = dspy.InputField(prefix="Discussiegeschiedenis: \n", format=str)
    information = dspy.InputField(prefix="Beschikbare informatie: \n", format=str)
    last_utterance = dspy.InputField(
        prefix="Laatste uitspraak in het gesprek: \n", format=str
    )
    output = dspy.OutputField(
        prefix="Geef nu de volgende discussiefocus in de vorm van een vraag van één zin:\n",
        format=str,
    )


class GroundedQuestionGenerationModule(dspy.Module):
    def __init__(self, engine: Union[dspy.dsp.LM, dspy.dsp.HFModel]):
        self.engine = engine
        self.gen_focus = dspy.Predict(GroundedQuestionGeneration)
        self.polish_style = dspy.Predict(ConvertUtteranceStyle)
        self.gen_summary = dspy.Predict(KnowledgeBaseSummmary)

    def forward(
        self,
        topic: str,
        knowledge_base: KnowledgeBase,
        last_conv_turn: ConversationTurn,
        unused_snippets: List[Information],
    ):
        # Formatteer ongebruikte informatiesnippers en maak een mapping van index naar informatie
        information, index_to_information_mapping = format_search_results(
            unused_snippets, info_max_num_words=1000
        )
        # Haal de samenvatting op van de kennisbank
        summary = knowledge_base.get_knowledge_base_summary()
        # Extraheer de laatste uitspraak zonder citaten
        last_utterance, _ = extract_and_remove_citations(last_conv_turn.utterance)
        
        with dspy.settings.context(lm=self.engine, show_guidelines=False):
            # Genereer een nieuwe discussiefocus
            raw_utterance = self.gen_focus(
                topic=topic,
                summary=summary,
                information=information,
                last_utterance=last_utterance,
            ).output
            # Verfijn de stijl van de gegenereerde uitspraak
            utterance = self.polish_style(
                expert="Moderator van het rondetafelgesprek",
                action="Een nieuwe vraag stellen door natuurlijk over te gaan van de vorige uitspraak.",
                prev=keep_first_and_last_paragraph(last_utterance),
                content=raw_utterance,
            ).utterance
            # Extraheer geciteerde zoekresultaten uit de verfijnde uitspraak
            cited_searched_results = extract_cited_storm_info(
                response=utterance, index_to_storm_info=index_to_information_mapping
            )
            # Retourneer de gegenereerde uitspraak met bijbehorende informatie
            return dspy.Prediction(
                raw_utterance=raw_utterance,
                utterance=utterance,
                cited_info=cited_searched_results,
            )