"""
Deze module initialiseert het Co-STORM systeem door een achtergrondonderzoek uit te voeren
om een gedeelde conceptuele ruimte met de gebruiker te creëren.

Deze fase functioneert als een mini-STORM, waarbij meerdere LLM-agenten met verschillende
perspectieven worden ingezet voor meerrondig overleg. De kennisbank (weergegeven als een
mindmap) wordt geïnitialiseerd met de informatie die tijdens deze uitwisselingen is verzameld.

Daarnaast genereert het systeem een eerste conceptrapport, dat vervolgens wordt gebruikt
om een beknopte en boeiende conversatie te creëren. Deze gesynthetiseerde conversatie
wordt aan de gebruiker gepresenteerd om hen snel bij te praten over de huidige kennis
van het systeem over het onderwerp.
"""

import dspy
import concurrent.futures
from threading import Lock
from typing import List, Optional, Union, TYPE_CHECKING

from .callback import BaseCallbackHandler
from .collaborative_storm_utils import _get_answer_question_module_instance
from .expert_generation import GenerateExpertModule
from .grounded_question_answering import AnswerQuestionModule
from ...dataclass import ConversationTurn, KnowledgeBase
from ...interface import LMConfigs
from ...logging_wrapper import LoggingWrapper
from ...storm_wiki.modules.outline_generation import WritePageOutline
from ...utils import ArticleTextProcessing as AP

if TYPE_CHECKING:
    from ..engine import RunnerArgument

class WarmStartModerator(dspy.Signature):
    """
    Je bent een moderator in een rondetafelgesprek. Het doel is om met meerdere experts te chatten
    om de feiten en achtergrond van het onderwerp te bespreken en het publiek vertrouwd te maken met het onderwerp.
    Je krijgt het onderwerp, de geschiedenis van vragen die je al hebt gesteld, en de huidige expert waarmee je in gesprek bent.
    Gebaseerd op deze informatie, genereer je de volgende vraag voor de huidige expert om de discussie verder te brengen.

    De output moet alleen de volgende vraag voor de huidige expert bevatten. Voeg geen andere informatie of inleiding toe.
    """

    topic = dspy.InputField(prefix="Onderwerp voor rondetafelgesprek: ", format=str)
    history = dspy.InputField(
        prefix="Experts waarmee je al hebt geïnteracteerd: ", format=str
    )
    current_expert = dspy.InputField(prefix="Expert waarmee je praat:", format=str)
    question = dspy.OutputField(
        prefix="Volgende vraag voor de expert waarmee je praat: ", format=str
    )

class SectionToConvTranscript(dspy.Signature):
    """
    Je krijgt een sectie van een kort rapport over een specifiek onderwerp. Je taak is om deze sectie
    om te zetten in een boeiende openingsdiscussie voor een rondetafelgesprek.
    Het doel is om deelnemers en het publiek snel de belangrijkste informatie te laten begrijpen.
    Zowel vraag als antwoord moeten in de toon van een rondetafelgesprek zijn, gericht op het publiek.

    Specifiek moet je:
    1. Een boeiende vraag genereren die gebruik maakt van de sectienaam en het onderwerp om de discussie over de inhoud te openen.
    2. Een kort en boeiend antwoord geven (met alle inline citaten uit de originele tekst) afgeleid van de sectie,
       die dient als aanknopingspunten en te veel details vermijdt.
    """

    topic = dspy.InputField(prefix="onderwerp:", format=str)
    section_name = dspy.InputField(prefix="sectienaam:", format=str)
    section_content = dspy.InputField(prefix="sectie-inhoud:", format=str)
    question = dspy.OutputField(prefix="Geef nu alleen een boeiende vraag.\nVraag:")
    answer = dspy.OutputField(
        prefix="Geef nu alleen een boeiend antwoord met alle inline citaten uit de originele tekst.\nAntwoord:"
    )

class ReportToConversation(dspy.Module):
    def __init__(self, engine: Union[dspy.dsp.LM, dspy.dsp.HFModel]):
        self.engine = engine
        self.section_to_conv_transcript = dspy.Predict(SectionToConvTranscript)

    def forward(self, knowledge_base: KnowledgeBase):
        def process_node(node, topic):
            # Verwerkt een enkele node om een vraag en antwoord te genereren
            with dspy.settings.context(lm=self.engine, show_guidelines=False):
                output = self.section_to_conv_transcript(
                    topic=topic,
                    section_name=node.get_path_from_root(),
                    section_content=node.synthesize_output,
                )
                question = output.question.replace("Vraag:", "").strip()
                answer = output.answer.replace("Antwoord:", "").strip()
                return question, answer

        conversations = []
        nodes = knowledge_base.collect_all_nodes()
        nodes = [node for node in nodes if node.name != "root" and node.content]
        topic = knowledge_base.topic

        # Gebruik multi-threading om de verwerking van nodes te versnellen
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future_to_node = {
                executor.submit(process_node, node, topic): node for node in nodes
            }
            for future in concurrent.futures.as_completed(future_to_node):
                node = future_to_node[future]
                question, answer = future.result()
                conversations.append(
                    ConversationTurn(
                        role="Achtergrondgesprek moderator",
                        raw_utterance=question,
                        utterance_type="Originele Vraag",
                        utterance=question,
                        cited_info=[
                            knowledge_base.info_uuid_to_info_dict[idx]
                            for idx in AP.parse_citation_indices(question)
                        ],
                    )
                )
                conversations.append(
                    ConversationTurn(
                        role="Achtergrondgesprek expert",
                        raw_utterance=answer,
                        utterance_type="Potentieel Antwoord",
                        utterance=answer,
                        cited_info=[
                            knowledge_base.info_uuid_to_info_dict[idx]
                            for idx in AP.parse_citation_indices(answer)
                        ],
                    )
                )
        return conversations

class WarmStartConversation(dspy.Module):
    def __init__(
        self,
        question_asking_lm: Union[dspy.dsp.LM, dspy.dsp.HFModel],
        generate_expert_module: GenerateExpertModule,
        answer_question_module: AnswerQuestionModule,
        logging_wrapper: LoggingWrapper,
        max_num_experts: int = 3,
        max_turn_per_experts: int = 2,
        max_thread: int = 3,
        callback_handler: BaseCallbackHandler = None,
    ):
        self.ask_question = dspy.Predict(WarmStartModerator)
        self.max_num_experts = max_num_experts
        self.max_turn_per_experts = max_turn_per_experts
        self.question_asking_lm = question_asking_lm
        self.answer_question_module = answer_question_module
        self.max_thread = max_thread
        self.generate_experts_module = generate_expert_module
        self.logging_wrapper = logging_wrapper
        self.callback_handler = callback_handler

    def format_dialogue_question_history_string(
        self, conversation_history: List[ConversationTurn]
    ):
        # Formatteert de gespreksgeschiedenis voor gebruik in volgende vragen
        output = []
        for idx, turn in enumerate(conversation_history):
            info = turn.claim_to_make if turn.claim_to_make else turn.utterance
            output.append(f"{idx + 1}: {info}")
        return "\n".join(output)

    def generate_warmstart_experts(self, topic: str):
        # Genereert experts en achtergrondinformatie voor het warmstart-proces
        background_seeking_dialogue = self.get_background_info(topic=topic)
        background_info = background_seeking_dialogue.utterance
        gen_expert_output = self.generate_experts_module(
            topic=topic,
            background_info=background_info,
            num_experts=self.max_num_experts,
        )
        return gen_expert_output.experts, background_seeking_dialogue

    def get_background_info(self, topic: str):
        # Haalt achtergrondinformatie op over het gegeven onderwerp
        question = f"Achtergrondinformatie over {topic}"
        answer = self.answer_question_module(
            topic=topic, question=question, mode="extensive", style="conversational"
        )

        return ConversationTurn(
            role="Standaard Achtergrondonderzoeker",
            raw_utterance=answer.response,
            utterance_type="Vragenstellen",
            claim_to_make=question,
            queries=answer.queries,
            raw_retrieved_info=answer.raw_retrieved_info,
            cited_info=answer.cited_info,
        )

    def forward(self, topic: str):
        with self.logging_wrapper.log_event(
            "warm start, perspectief-geleide VenA: experts identificeren"
        ):
            # Voer achtergrondonderzoek uit, genereer enkele experts
            experts, background_seeking_dialogue = self.generate_warmstart_experts(
                topic=topic
            )
        # Initialiseer lijst om de gespreksgeschiedenis op te slaan
        conversation_history: List[ConversationTurn] = []
        lock = Lock()

        # Hiërarchische chat: chat met één expert. Genereer vraag, krijg antwoord
        def process_expert(expert):
            expert_name, expert_description = expert.split(":")
            for idx in range(self.max_turn_per_experts):
                with self.logging_wrapper.log_event(
                    f"warm start, perspectief-geleide VenA: expert {expert_name}; beurt {idx + 1}"
                ):
                    try:
                        with lock:
                            history = self.format_dialogue_question_history_string(
                                conversation_history
                            )
                        with dspy.settings.context(lm=self.question_asking_lm):
                            question = self.ask_question(
                                topic=topic, history=history, current_expert=expert
                            ).question
                        answer = self.answer_question_module(
                            topic=topic,
                            question=question,
                            mode="brief",
                            style="conversational",
                        )
                        conversation_turn = ConversationTurn(
                            role=expert,
                            claim_to_make=question,
                            raw_utterance=answer.response,
                            utterance_type="Ondersteuning",
                            queries=answer.queries,
                            raw_retrieved_info=answer.raw_retrieved_info,
                            cited_info=answer.cited_info,
                        )
                        if self.callback_handler is not None:
                            self.callback_handler.on_warmstart_update(
                                message="\n".join(
                                    [
                                        f"Klaar met browsen {url}"
                                        for url in [
                                            i.url for i in answer.raw_retrieved_info
                                        ]
                                    ]
                                )
                            )
                        with lock:
                            conversation_history.append(conversation_turn)
                    except Exception as e:
                        print(f"Fout bij het verwerken van expert {expert}: {e}")

        # Multi-thread conversatie
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=self.max_thread
        ) as executor:
            futures = [
                executor.submit(process_expert, expert)
                for expert in experts[: min(len(experts), self.max_num_experts)]
            ]
            concurrent.futures.wait(futures)

        conversation_history = [background_seeking_dialogue] + conversation_history

        return dspy.Prediction(
            conversation_history=conversation_history, experts=experts
        )

class GenerateWarmStartOutline(dspy.Signature):
    """Genereer een outline van het wikipedia-achtige rapport van een rondetafelgesprek. Je krijgt discussiepunten
    uit het gesprek en bijbehorende zoekopdrachten te zien.
    Je krijgt een conceptoutline waaruit je inspiratie kunt putten. Neem geen secties op die niet genoemd worden
    in de gegeven gespreksgeschiedenis.
    Gebruik "#" om sectiekoppen aan te geven, "##" voor subsectiekoppen, enzovoort.
     Volg deze richtlijnen:
     1. Gebruik "#" voor sectietitels, "##" voor subsectietitels, "###" voor subsubsectietitels, enzovoort.
     2. Voeg geen aanvullende informatie toe.
     3. Sluit de onderwerpnaam uit van de outline.
     De organisatie van de outline moet de Wikipedia-stijl aannemen.
    """

    topic = dspy.InputField(prefix="Het besproken onderwerp: ", format=str)
    draft = dspy.InputField(prefix="Conceptoutline waar je naar kunt verwijzen: ", format=str)
    conv = dspy.InputField(prefix="Gespreksgeschiedenis:\n", format=str)
    outline = dspy.OutputField(
        prefix='Schrijf de gespreksoutline (Gebruik "#" Titel" om een sectietitel aan te geven, "##" Titel" om een subsectietitel aan te geven, ...):\n',
        format=str,
    )

class GenerateWarmStartOutlineModule(dspy.Module):
    def __init__(self, engine: Union[dspy.dsp.LM, dspy.dsp.HFModel]):
        self.engine = engine
        self.gen_outline = dspy.Predict(GenerateWarmStartOutline)
        self.draft_outline = dspy.Predict(WritePageOutline)

    def extract_questions_and_queries(self, conv: List[ConversationTurn]):
        # Extraheert vragen en zoekopdrachten uit de gespreksgeschiedenis
        context = []
        for turn in conv:
            focus = turn.claim_to_make
            queries = turn.queries
            queries_string = "\n\t".join(
                f"Zoekopdracht {idx + 1}: {query}" for idx, query in enumerate(queries)
            )
            string = f"Discussiefocus {len(context) + 1}: {focus}\n\t{queries_string}"
            context.append(string)
        return "\n".join(context)

    def get_draft_outline(self, topic: str):
        # Genereert een conceptoutline voor het gegeven onderwerp
        with dspy.settings.context(lm=self.engine):
            return self.draft_outline(topic=topic).outline

    def forward(self, topic: str, conv: List[ConversationTurn]):
        discussion_history = self.extract_questions_and_queries(conv)
        draft_outline = self.get_draft_outline(topic=topic)
        with dspy.settings.context(lm=self.engine):
            outline = self.gen_outline(
                topic=topic, draft=draft_outline, conv=discussion_history
            ).outline
            outline = AP.clean_up_outline(outline)
        return dspy.Prediction(outline=outline, draft_outline=draft_outline)

class WarmStartModule:
    def __init__(
        self,
        lm_config: LMConfigs,
        runner_argument: "RunnerArgument",
        logging_wrapper: LoggingWrapper,
        rm: Optional[dspy.Retrieve] = None,
        callback_handler: BaseCallbackHandler = None,
    ):
        generate_expert_module = GenerateExpertModule(
            engine=lm_config.discourse_manage_lm
        )
        self.warmstart_conv = WarmStartConversation(
            question_asking_lm=lm_config.question_asking_lm,
            generate_expert_module=generate_expert_module,
            answer_question_module=_get_answer_question_module_instance(
                lm_config=lm_config,
                runner_argument=runner_argument,
                logging_wrapper=logging_wrapper,
                rm=rm,
            ),
            max_num_experts=runner_argument.warmstart_max_num_experts,
            max_turn_per_experts=runner_argument.warmstart_max_turn_per_experts,
            max_thread=runner_argument.warmstart_max_thread,
            logging_wrapper=logging_wrapper,
            callback_handler=callback_handler,
        )
        self.warmstart_outline_gen_module = GenerateWarmStartOutlineModule(
            engine=lm_config.warmstart_outline_gen_lm
        )
        self.report_to_conversation = ReportToConversation(lm_config.knowledge_base_lm)
        self.logging_wrapper = logging_wrapper
        self.callback_handler = callback_handler

    def initiate_warm_start(self, topic: str, knowledge_base: KnowledgeBase):
        """
        Initieert een warm start-proces voor het gegeven onderwerp door een warm start-gesprek te genereren en
        de resulterende informatie in een kennisbank in te voegen.

        Args:
            topic (str): Het onderwerp waarvoor het warm start-proces moet worden geïnitieerd.

        Returns:
            Tuple[List[ConversationTurn], List[str], KnowledgeBase]:
                - Een lijst van ConversationTurn-instanties die de gespreksgeschiedenis vertegenwoordigen.
                - Een lijst van strings die de experts vertegenwoordigen die bij het gesprek betrokken zijn.
                - Een KnowledgeBase-instantie die de georganiseerde informatie bevat.
        """
        warm_start_conversation_history: List[ConversationTurn] = []
        warm_start_experts = None
        # Haal warm start-gesprekken op
        with self.logging_wrapper.log_event("warm start: perspectief-geleide VenA"):
            if self.callback_handler is not None:
                self.callback_handler.on_warmstart_update(
                    message="Start met vertrouwd raken met het onderwerp door te chatten met meerdere LLM-experts (Stap 1 / 4)"
                )
            warm_start_result = self.warmstart_conv(topic=topic)
            warm_start_conversation_history = warm_start_result.conversation_history
            warm_start_experts = warm_start_result.experts

        # Haal warm start-gespreksoutline op
        with self.logging_wrapper.log_event("warm start: outline generatie"):
            if self.callback_handler is not None:
                self.callback_handler.on_warmstart_update(
                    "Verzamelde informatie organiseren (Stap 2 / 4)"
                )
            warm_start_outline_output = self.warmstart_outline_gen_module(
                topic=topic, conv=warm_start_conversation_history
            )
        # Initialiseer kennisbank
        with self.logging_wrapper.log_event("warm start: invoegen in kennisbank"):
            if self.callback_handler is not None:
                self.callback_handler.on_warmstart_update(
                    "Verzamelde informatie invoegen in kennisbank (Stap 3 / 4)"
                )
            knowledge_base.insert_from_outline_string(
                outline_string=warm_start_outline_output.outline
            )
            # Voeg informatie in de kennisbank in
            for turn in warm_start_conversation_history:
                knowledge_base.update_from_conv_turn(
                    conv_turn=turn, allow_create_new_node=False
                )
        # Kennisbank naar rapport
        if self.callback_handler is not None:
            self.callback_handler.on_warmstart_update(
                "Achtergrondinformatie discussie-uitspraken synthetiseren (Stap 4 / 4)"
            )
        knowledge_base.to_report()

        # Genereer boeiende gesprekken
        engaging_conversations = self.report_to_conversation(knowledge_base)
        return (
            warm_start_conversation_history,
            engaging_conversations,
            warm_start_experts,
        )