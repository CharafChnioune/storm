import concurrent.futures
import logging
import os
from concurrent.futures import as_completed
from typing import Union, List, Tuple, Optional, Dict

import dspy

from .callback import BaseCallbackHandler
from .persona_generator import StormPersonaGenerator
from .storm_dataclass import DialogueTurn, StormInformationTable
from ...interface import KnowledgeCurationModule, Retriever, Information
from ...utils import ArticleTextProcessing

# Controleer of Streamlit beschikbaar is voor logging context
try:
    from streamlit.runtime.scriptrunner import add_script_run_ctx
    streamlit_connection = True
except ImportError as err:
    streamlit_connection = False

script_dir = os.path.dirname(os.path.abspath(__file__))

class ConvSimulator(dspy.Module):
    """
    Simuleert een gesprek tussen een Wikipedia-schrijver met een specifieke persona en een expert.
    
    Deze klasse gebruikt twee taalmodellen: één voor de expert en één voor de vraagsteller,
    en een retriever om relevante informatie op te halen tijdens het gesprek.
    """

    def __init__(
        self,
        topic_expert_engine: Union[dspy.dsp.LM, dspy.dsp.HFModel],
        question_asker_engine: Union[dspy.dsp.LM, dspy.dsp.HFModel],
        retriever: Retriever,
        max_search_queries_per_turn: int,
        search_top_k: int,
        max_turn: int,
    ):
        super().__init__()
        # Initialiseer componenten voor de simulatie
        self.wiki_writer = WikiWriter(engine=question_asker_engine)
        self.topic_expert = TopicExpert(
            engine=topic_expert_engine,
            max_search_queries=max_search_queries_per_turn,
            search_top_k=search_top_k,
            retriever=retriever,
        )
        self.max_turn = max_turn

    def forward(
        self,
        topic: str,
        persona: str,
        ground_truth_url: str,
        callback_handler: BaseCallbackHandler,
    ):
        """
        Voert de conversatiesimulatie uit.

        Args:
            topic: Het onderwerp waarover onderzoek wordt gedaan.
            persona: De persona van de Wikipedia-schrijver.
            ground_truth_url: Deze URL wordt uitgesloten van de zoekopdracht om lekkage van grondwaarheid bij evaluatie te voorkomen.
            callback_handler: Handler voor callbacks tijdens de simulatie.

        Returns:
            Een dspy.Prediction object met de gespreksgeschiedenis.
        """
        dlg_history: List[DialogueTurn] = []
        for _ in range(self.max_turn):
            # Genereer een vraag van de Wikipedia-schrijver
            user_utterance = self.wiki_writer(
                topic=topic, persona=persona, dialogue_turns=dlg_history
            ).question
            if user_utterance == "":
                logging.error("Gesimuleerde Wikipedia-schrijver uiting is leeg.")
                break
            if user_utterance.startswith("Hartelijk dank voor je hulp!"):
                break
            # Genereer een antwoord van de expert
            expert_output = self.topic_expert(
                topic=topic, question=user_utterance, ground_truth_url=ground_truth_url
            )
            dlg_turn = DialogueTurn(
                agent_utterance=expert_output.answer,
                user_utterance=user_utterance,
                search_queries=expert_output.queries,
                search_results=expert_output.searched_results,
            )
            dlg_history.append(dlg_turn)
            callback_handler.on_dialogue_turn_end(dlg_turn=dlg_turn)

        return dspy.Prediction(dlg_history=dlg_history)


class WikiWriter(dspy.Module):
    """
    Genereert vragen vanuit het perspectief van een Wikipedia-schrijver in een conversationele setup.
    
    De gegenereerde vraag wordt gebruikt om een nieuwe ronde van informatieverzameling te starten.
    """

    def __init__(self, engine: Union[dspy.dsp.LM, dspy.dsp.HFModel]):
        super().__init__()
        self.ask_question_with_persona = dspy.ChainOfThought(AskQuestionWithPersona)
        self.ask_question = dspy.ChainOfThought(AskQuestion)
        self.engine = engine

    def forward(
        self,
        topic: str,
        persona: str,
        dialogue_turns: List[DialogueTurn],
        draft_page=None,
    ):
        # Bereid de conversatiegeschiedenis voor
        conv = []
        for turn in dialogue_turns[:-4]:
            conv.append(
                f"You: {turn.user_utterance}\nExpert: Omit the answer here due to space limit."
            )
        for turn in dialogue_turns[-4:]:
            conv.append(
                f"You: {turn.user_utterance}\nExpert: {ArticleTextProcessing.remove_citations(turn.agent_utterance)}"
            )
        conv = "\n".join(conv)
        conv = conv.strip() or "N/A"
        conv = ArticleTextProcessing.limit_word_count_preserve_newline(conv, 2500)

        # Genereer een vraag met of zonder persona
        with dspy.settings.context(lm=self.engine):
            if persona is not None and len(persona.strip()) > 0:
                question = self.ask_question_with_persona(
                    topic=topic, persona=persona, conv=conv
                ).question
            else:
                question = self.ask_question(
                    topic=topic, persona=persona, conv=conv
                ).question

        return dspy.Prediction(question=question)


class AskQuestion(dspy.Signature):
    """
    Je bent een ervaren Wikipedia-schrijver. Je chat met een expert om informatie te krijgen voor het onderwerp waarover je wilt bijdragen. 
    Stel goede vragen om meer nuttige informatie te krijgen die relevant is voor het onderwerp.
    Als je geen vragen meer hebt, zeg dan "Hartelijk dank voor je hulp!" om het gesprek te beëindigen.
    Stel alsjeblieft één vraag tegelijk en herhaal geen vragen die je al eerder hebt gesteld. Je vragen moeten gerelateerd zijn aan het onderwerp waarover je wilt schrijven.
    """

    topic = dspy.InputField(prefix="Onderwerp waarover je wilt schrijven: ", format=str)
    conv = dspy.InputField(prefix="Gespreksgeschiedenis:\n", format=str)
    question = dspy.OutputField(format=str)


class AskQuestionWithPersona(dspy.Signature):
    """
    Je bent een ervaren Wikipedia-schrijver en wilt een specifieke pagina bewerken. Naast je identiteit als Wikipedia-schrijver heb je een specifieke focus bij het onderzoeken van het onderwerp.
    Nu chat je met een expert om informatie te krijgen. Stel goede vragen om meer nuttige informatie te krijgen.
    Als je geen vragen meer hebt, zeg dan "Hartelijk dank voor je hulp!" om het gesprek te beëindigen.
    Stel alsjeblieft één vraag tegelijk en herhaal geen vragen die je al eerder hebt gesteld. Je vragen moeten gerelateerd zijn aan het onderwerp waarover je wilt schrijven.
    """

    topic = dspy.InputField(prefix="Onderwerp waarover je wilt schrijven: ", format=str)
    persona = dspy.InputField(
        prefix="Je persona naast het zijn van een Wikipedia-schrijver: ", format=str
    )
    conv = dspy.InputField(prefix="Gespreksgeschiedenis:\n", format=str)
    question = dspy.OutputField(format=str)


class QuestionToQuery(dspy.Signature):
    """
    Je wilt de vraag beantwoorden met behulp van Google-zoekopdrachten. Wat typ je in het zoekvak?
    Schrijf de zoekopdrachten die je zult gebruiken in het volgende formaat:
    - zoekopdracht 1
    - zoekopdracht 2
    ...
    - zoekopdracht n
    """

    topic = dspy.InputField(prefix="Onderwerp waarover je discussieert: ", format=str)
    question = dspy.InputField(prefix="Vraag die je wilt beantwoorden: ", format=str)
    queries = dspy.OutputField(format=str)


class AnswerQuestion(dspy.Signature):
    """
    Je bent een expert die effectief informatie kan gebruiken. Je chat met een Wikipedia-schrijver die een Wikipedia-pagina wil schrijven over een onderwerp dat je kent. Je hebt gerelateerde informatie verzameld en zult deze nu gebruiken om een antwoord te formuleren.
    Maak je antwoord zo informatief mogelijk en zorg ervoor dat elke zin wordt ondersteund door de verzamelde informatie. Als de [verzamelde informatie] niet direct gerelateerd is aan het [onderwerp] of de [vraag], geef dan het meest relevante antwoord op basis van de beschikbare informatie. Als er geen passend antwoord kan worden geformuleerd, antwoord dan met "Ik kan deze vraag niet beantwoorden op basis van de beschikbare informatie" en leg eventuele beperkingen of hiaten uit.
    """

    topic = dspy.InputField(prefix="Onderwerp waarover je discussieert:", format=str)
    conv = dspy.InputField(prefix="Vraag:\n", format=str)
    info = dspy.InputField(prefix="Verzamelde informatie:\n", format=str)
    answer = dspy.OutputField(
        prefix="Geef nu je antwoord. (Probeer zoveel mogelijk verschillende bronnen te gebruiken en voeg geen verzonnen informatie toe.)\n",
        format=str,
    )


class TopicExpert(dspy.Module):
    """
    Beantwoordt vragen met behulp van op zoekopdrachten gebaseerde retrieval en antwoordgeneratie. Deze module voert de volgende stappen uit:
    1. Genereer zoekopdrachten op basis van de vraag.
    2. Zoek naar informatie met behulp van de zoekopdrachten.
    3. Filter onbetrouwbare bronnen uit.
    4. Genereer een antwoord met behulp van de opgehaalde informatie.
    """

    def __init__(
        self,
        engine: Union[dspy.dsp.LM, dspy.dsp.HFModel],
        max_search_queries: int,
        search_top_k: int,
        retriever: Retriever,
    ):
        super().__init__()
        self.generate_queries = dspy.Predict(QuestionToQuery)
        self.retriever = retriever
        self.answer_question = dspy.Predict(AnswerQuestion)
        self.engine = engine
        self.max_search_queries = max_search_queries
        self.search_top_k = search_top_k

    def forward(self, topic: str, question: str, ground_truth_url: str):
        with dspy.settings.context(lm=self.engine, show_guidelines=False):
            # Identificeer: Splits de vraag op in zoekopdrachten
            queries = self.generate_queries(topic=topic, question=question).queries
            queries = [
                q.replace("-", "").strip().strip('"').strip('"').strip()
                for q in queries.split("\n")
            ]
            queries = queries[: self.max_search_queries]
            # Zoek
            searched_results: List[Information] = self.retriever.retrieve(
                list(set(queries)), exclude_urls=[ground_truth_url]
            )
            if len(searched_results) > 0:
                # Evalueer: Vereenvoudig dit deel door direct het bovenste snippet te gebruiken
                info = ""
                for n, r in enumerate(searched_results):
                    info += "\n".join(f"[{n + 1}]: {s}" for s in r.snippets[:1])
                    info += "\n\n"

                info = ArticleTextProcessing.limit_word_count_preserve_newline(
                    info, 1000
                )

                try:
                    answer = self.answer_question(
                        topic=topic, conv=question, info=info
                    ).answer
                    answer = ArticleTextProcessing.remove_uncompleted_sentences_with_citations(
                        answer
                    )
                except Exception as e:
                    logging.error(f"Fout bij het genereren van antwoord: {e}")
                    answer = "Sorry, ik kan deze vraag niet beantwoorden. Stel alsjeblieft een andere vraag."
            else:
                # Als er geen informatie wordt gevonden, moet de expert niet fantaseren
                answer = "Sorry, ik kan geen informatie vinden voor deze vraag. Stel alsjeblieft een andere vraag."

        return dspy.Prediction(
            queries=queries, searched_results=searched_results, answer=answer
        )


class StormKnowledgeCurationModule(KnowledgeCurationModule):
    """
    De interface voor de kenniscuratiefase. Gegeven een onderwerp, retourneert deze module verzamelde informatie.
    
    Deze module coördineert het proces van kenniscuratie, inclusief het genereren van persona's,
    het simuleren van gesprekken, en het verzamelen van informatie uit deze gesprekken.
    """

    def __init__(
        self,
        retriever: Retriever,
        persona_generator: Optional[StormPersonaGenerator],
        conv_simulator_lm: Union[dspy.dsp.LM, dspy.dsp.HFModel],
        question_asker_lm: Union[dspy.dsp.LM, dspy.dsp.HFModel],
        max_search_queries_per_turn: int,
        search_top_k: int,
        max_conv_turn: int,
        max_thread_num: int,
    ):
        """
        Initialiseert de module met de benodigde componenten en parameters.
        """
        self.retriever = retriever
        self.persona_generator = persona_generator
        self.conv_simulator_lm = conv_simulator_lm
        self.search_top_k = search_top_k
        self.max_thread_num = max_thread_num
        self.retriever = retriever
        self.conv_simulator = ConvSimulator(
            topic_expert_engine=conv_simulator_lm,
            question_asker_engine=question_asker_lm,
            retriever=retriever,
            max_search_queries_per_turn=max_search_queries_per_turn,
            search_top_k=search_top_k,
            max_turn=max_conv_turn,
        )

    def _get_considered_personas(self, topic: str, max_num_persona) -> List[str]:
        """
        Genereert een lijst van persona's voor het gegeven onderwerp.
        """
        return self.persona_generator.generate_persona(
            topic=topic, max_num_persona=max_num_persona
        )

    def _run_conversation(
        self,
        conv_simulator,
        topic,
        ground_truth_url,
        considered_personas,
        callback_handler: BaseCallbackHandler,
    ) -> List[Tuple[str, List[DialogueTurn]]]:
        """
        Voert meerdere gesprekssimulaties parallel uit, elk met een andere persona,
        en verzamelt hun gespreksgeschiedenissen.

        Returns:
            Een lijst van tuples, waarbij elke tuple een persona en de bijbehorende
            opgeschoonde gespreksgeschiedenis bevat.
        """
        conversations = []

        def run_conv(persona):
            return conv_simulator(
                topic=topic,
                ground_truth_url=ground_truth_url,
                persona=persona,
                callback_handler=callback_handler,
            )

        max_workers = min(self.max_thread_num, len(considered_personas))

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_persona = {
                executor.submit(run_conv, persona): persona
                for persona in considered_personas
            }

            if streamlit_connection:
                # Zorg voor de juiste logging context bij verbinding met Streamlit frontend
                for t in executor._threads:
                    add_script_run_ctx(t)

            for future in as_completed(future_to_persona):
                persona = future_to_persona[future]
                conv = future.result()
                conversations.append(
                    (persona, ArticleTextProcessing.clean_up_citation(conv).dlg_history)
                )

        return conversations

    def research(
        self,
        topic: str,
        ground_truth_url: str,
        callback_handler: BaseCallbackHandler,
        max_perspective: int = 0,
        disable_perspective: bool = True,
        return_conversation_log=False,
    ) -> Union[StormInformationTable, Tuple[StormInformationTable, Dict]]:
        """
        Verzamelt informatie en kennis voor het gegeven onderwerp.

        Args:
            topic: Onderwerp van interesse in natuurlijke taal.
            ground_truth_url: URL van de grondwaarheid, uitgesloten van zoekopdrachten.
            callback_handler: Handler voor callbacks tijdens het onderzoeksproces.
            max_perspective: Maximaal aantal te genereren persona's.
            disable_perspective: Indien True, worden geen persona's gebruikt.
            return_conversation_log: Indien True, wordt ook een log van de gesprekken geretourneerd.

        Returns:
            Een StormInformationTable met verzamelde informatie, en optioneel een log van de gesprekken.
        """
        # Identificeer persona's
        callback_handler.on_identify_perspective_start()
        considered_personas = []
        if disable_perspective:
            considered_personas = [""]
        else:
            considered_personas = self._get_considered_personas(
                topic=topic, max_num_persona=max_perspective
            )
        callback_handler.on_identify_perspective_end(perspectives=considered_personas)

        # Voer gesprekken
        callback_handler.on_information_gathering_start()
        conversations = self._run_conversation(
            conv_simulator=self.conv_simulator,
            topic=topic,
            ground_truth_url=ground_truth_url,
            considered_personas=considered_personas,
            callback_handler=callback_handler,
        )

        information_table = StormInformationTable(conversations)
        callback_handler.on_information_gathering_end()
        
        if return_conversation_log:
            return information_table, StormInformationTable.construct_log_dict(
                conversations
            )
        return information_table