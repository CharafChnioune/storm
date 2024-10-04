# Importeer dspy voor AI-gestuurde modules en signatures
import dspy
from typing import Union

# Importeer hulpfuncties voor tekstverwerking en citaatbeheer
from .collaborative_storm_utils import (
    trim_output_after_hint,
    extract_and_remove_citations,
    keep_first_and_last_paragraph,
)

# Importeer modules voor vraagbeantwoording en stijlaanpassing
from .grounded_question_answering import AnswerQuestionModule
from .grounded_question_generation import ConvertUtteranceStyle

# Importeer datastructuren en logging functionaliteit
from ...dataclass import ConversationTurn
from ...logging_wrapper import LoggingWrapper
from .callback import BaseCallbackHandler

class GenExpertActionPlanning(dspy.Signature):
    """
    Definieert de structuur voor het plannen van acties voor een expert in een rondetafelgesprek.
    
    Doel: Bepaal de volgende actie van een expert op basis van de gesprekscontext.

    Aannames:
    - Expert heeft toegang tot volledige gespreksgeschiedenis
    - Keuze uit vier voorgedefinieerde actietypes

    Beperking: Output moet strikt het gespecificeerde formaat volgen
    """

    # Inputvelden voor de expert actie planning
    topic = dspy.InputField(prefix="Gespreksonderwerp: ", format=str)
    expert = dspy.InputField(prefix="U bent uitgenodigd als: ", format=str)
    summary = dspy.InputField(prefix="Gespreksgeschiedenis: \n", format=str)
    last_utterance = dspy.InputField(
        prefix="Laatste uitspraak in het gesprek: \n", format=str
    )
    resposne = dspy.OutputField(
        prefix="Geef nu uw notitie. Begin met een van [Originele Vraag, Verdere Details, Informatieverzoek, Potentieel Antwoord] gevolgd door een beschrijving in één zin\n",
        format=str,
    )


class CoStormExpertUtteranceGenerationModule(dspy.Module):
    def __init__(
        self,
        action_planning_lm: Union[dspy.dsp.LM, dspy.dsp.HFModel],
        utterance_polishing_lm: Union[dspy.dsp.LM, dspy.dsp.HFModel],
        answer_question_module: AnswerQuestionModule,
        logging_wrapper: LoggingWrapper,
        callback_handler: BaseCallbackHandler = None,
    ):
        """
        Initialiseert de module voor het genereren van expertuitspraken in een CoStorm-sessie.
        
        Coördineert actieplanning, antwoordgeneratie en stijlverfijning voor experts.
        """
        self.action_planning_lm = action_planning_lm
        self.utterance_polishing_lm = utterance_polishing_lm
        self.expert_action = dspy.Predict(GenExpertActionPlanning)
        self.change_style = dspy.Predict(ConvertUtteranceStyle)
        self.answer_question_module = answer_question_module
        self.logging_wrapper = logging_wrapper
        self.callback_handler = callback_handler

    def parse_action(self, action):
        """
        Extraheert actietype en inhoud uit de expertactie.
        
        Zoekt naar voorgedefinieerde actietypes en splitst de actie op.
        """
        action_types = [
            "Originele Vraag",
            "Verdere Details",
            "Informatieverzoek",
            "Potentieel Antwoord",
        ]
        for action_type in action_types:
            if f"{action_type}:" in action:
                return action_type, trim_output_after_hint(action, f"{action_type}:")
            elif f"[{action_type}]:" in action:
                return action_type, trim_output_after_hint(action, f"[{action_type}]:")
        # Randgeval: geen geldig actietype gevonden
        return "Ongedefinieerd", ""

    def polish_utterance(
        self, conversation_turn: ConversationTurn, last_conv_turn: ConversationTurn
    ):
        """
        Verfijnt de stijl van de expertuitspraak.
        
        Gebruikt utterance_polishing_lm om de uitspraak aan te passen aan de gesprekscontext.
        """
        action_type = conversation_turn.utterance_type
        with self.logging_wrapper.log_event(
            "RoundTableConversationModule.ConvertUtteranceStyle"
        ):
            with dspy.settings.context(
                lm=self.utterance_polishing_lm, show_guidelines=False
            ):
                # Bereid actiestring voor op basis van actietype
                action_string = (
                    f"{action_type} over: {conversation_turn.claim_to_make}"
                    if action_type not in ["Originele Vraag", "Informatieverzoek"]
                    else f"{action_type}"
                )
                
                # Verwijder citaten en trim laatste uitspraak voor context
                last_expert_utterance_wo_citation, _ = extract_and_remove_citations(
                    last_conv_turn.utterance
                )
                trimmed_last_expert_utterance = keep_first_and_last_paragraph(
                    last_expert_utterance_wo_citation
                )
                
                # Pas uitspraakstijl aan met taalmodel
                utterance = self.change_style(
                    expert=conversation_turn.role,
                    action=action_string,
                    prev=trimmed_last_expert_utterance,
                    content=conversation_turn.raw_utterance,
                ).utterance
            conversation_turn.utterance = utterance

    def forward(
        self,
        topic: str,
        current_expert: str,
        conversation_summary: str,
        last_conv_turn: ConversationTurn,
    ):
        """
        Genereert een nieuwe expertuitspraak op basis van de gesprekscontext.
        
        Coördineert actieplanning, antwoordgeneratie en stijlverfijning.
        """
        # Verwijder citaten uit laatste uitspraak voor schone context
        last_utterance, _ = extract_and_remove_citations(last_conv_turn.utterance)
        
        # Bepaal actietype op basis van vorige uitspraak
        if last_conv_turn.utterance_type in ["Originele Vraag", "Informatieverzoek"]:
            action_type = "Potentieel Antwoord"
            action_content = last_utterance
        else:
            # Plan expertactie met action_planning_lm
            with self.logging_wrapper.log_event(
                "CoStormExpertUtteranceGenerationModule: GenExpertActionPlanning"
            ):
                with dspy.settings.context(
                    lm=self.action_planning_lm, show_guidelines=False
                ):
                    action = self.expert_action(
                        topic=topic,
                        expert=current_expert,
                        summary=conversation_summary,
                        last_utterance=last_utterance,
                    ).resposne
                action_type, action_content = self.parse_action(action)

        # Roep callback aan indien ingesteld (voor externe integratie)
        if self.callback_handler is not None:
            self.callback_handler.on_expert_action_planning_end()
        
        # Initialiseer nieuwe gespreksturn
        conversation_turn = ConversationTurn(
            role=current_expert, raw_utterance="", utterance_type=action_type
        )

        # Verwerk actie op basis van actietype
        if action_type == "Ongedefinieerd":
            raise Exception(f"Onverwachte output: {action}")
        elif action_type in ["Verdere Details", "Potentieel Antwoord"]:
            # Genereer gedetailleerd antwoord met answer_question_module
            with self.logging_wrapper.log_event(
                "RoundTableConversationModule: QuestionAnswering"
            ):
                grounded_answer = self.answer_question_module(
                    topic=topic,
                    question=action_content,
                    mode="kort",
                    style="conversationeel en beknopt",
                    callback_handler=self.callback_handler,
                )
            # Vul conversation_turn met gegenereerde informatie
            conversation_turn.claim_to_make = action_content
            conversation_turn.raw_utterance = grounded_answer.response
            conversation_turn.queries = grounded_answer.queries
            conversation_turn.raw_retrieved_info = grounded_answer.raw_retrieved_info
            conversation_turn.cited_info = grounded_answer.cited_info
        elif action_type in ["Originele Vraag", "Informatieverzoek"]:
            # Gebruik gegenereerde vraag of informatieverzoek direct
            conversation_turn.raw_utterance = action_content

        return dspy.Prediction(conversation_turn=conversation_turn)