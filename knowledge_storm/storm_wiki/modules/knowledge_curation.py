import concurrent.futures
import logging
import os
from concurrent.futures import as_completed
from typing import Union, List, Tuple, Optional, Dict

import dspy

from .callback import BaseCallbackHandler
from .persona_generator import StormPersonaGenerator
from .storm_dataclass import DialogueTurn, StormInformationTable, StormInformation
from ...interface import KnowledgeCurationModule, Retriever, CombinedRetriever
from ...utils import ArticleTextProcessing
from .retriever import YouRMRetriever, VectorRMRetriever

try:
    from streamlit.runtime.scriptrunner import add_script_run_ctx
    streamlit_connection = True
except ImportError as err:
    streamlit_connection = False

logger = logging.getLogger(__name__)

class ConvSimulator(dspy.Module):
    """Simulate a conversation between a Wikipedia writer with specific persona and an expert."""

    def __init__(self, topic_expert_engine: Union[dspy.dsp.LM, dspy.dsp.HFModel],
                 question_asker_engine: Union[dspy.dsp.LM, dspy.dsp.HFModel],
                 retriever: Retriever, max_search_queries_per_turn: int, search_top_k: int, max_turn: int):
        super().__init__()
        self.wiki_writer = WikiWriter(engine=question_asker_engine)
        self.topic_expert = TopicExpert(
            engine=topic_expert_engine,
            max_search_queries=max_search_queries_per_turn,
            search_top_k=search_top_k,
            retriever=retriever
        )
        self.max_turn = max_turn

    def forward(self, topic: str, persona: str, ground_truth_url: str, callback_handler: BaseCallbackHandler):
        """
        topic: The topic to research.
        persona: The persona of the Wikipedia writer.
        ground_truth_url: The ground_truth_url will be excluded from search to avoid ground truth leakage in evaluation.
        """
        dlg_history: List[DialogueTurn] = []
        for _ in range(self.max_turn):
            user_utterance = self.wiki_writer(topic=topic, persona=persona, dialogue_turns=dlg_history).question
            if user_utterance == '':
                logging.error('Simulated Wikipedia writer utterance is empty.')
                break
            if user_utterance.startswith('Thank you so much for your help!'):
                break
            expert_output = self.topic_expert(topic=topic, question=user_utterance, ground_truth_url=ground_truth_url)
            dlg_turn = DialogueTurn(
                agent_utterance=expert_output.answer,
                user_utterance=user_utterance,
                search_queries=expert_output.queries,
                search_results=expert_output.searched_results
            )
            dlg_history.append(dlg_turn)
            callback_handler.on_dialogue_turn_end(dlg_turn=dlg_turn)

        return dspy.Prediction(dlg_history=dlg_history)

class WikiWriter(dspy.Module):
    """Perspective-guided question asking in conversational setup.

    The asked question will be used to start a next round of information seeking."""

    def __init__(self, engine: Union[dspy.dsp.LM, dspy.dsp.HFModel]):
        super().__init__()
        self.ask_question_with_persona = dspy.ChainOfThought(AskQuestionWithPersona)
        self.ask_question = dspy.ChainOfThought(AskQuestion)
        self.engine = engine

    def forward(self, topic: str, persona: str, dialogue_turns: List[DialogueTurn], draft_page=None):
        conv = []
        for turn in dialogue_turns[:-4]:
            conv.append(f'You: {turn.user_utterance}\nExpert: Omit the answer here due to space limit.')
        for turn in dialogue_turns[-4:]:
            conv.append(
                f'You: {turn.user_utterance}\nExpert: {ArticleTextProcessing.remove_citations(turn.agent_utterance)}')
        conv = '\n'.join(conv)
        conv = conv.strip() or 'N/A'
        conv = ArticleTextProcessing.limit_word_count_preserve_newline(conv, 2500)

        with dspy.settings.context(lm=self.engine):
            if persona is not None and len(persona.strip()) > 0:
                question = self.ask_question_with_persona(topic=topic, persona=persona, conv=conv).question
            else:
                question = self.ask_question(topic=topic, persona=persona, conv=conv).question

        return dspy.Prediction(question=question)

class AskQuestion(dspy.Signature):
    """You are an experienced Wikipedia writer. You are chatting with an expert to get information for the topic you want to contribute. Ask good questions to get more useful information relevant to the topic.
    When you have no more question to ask, say "Thank you so much for your help!" to end the conversation.
    Please only ask a question at a time and don't ask what you have asked before. Your questions should be related to the topic you want to write."""

    topic = dspy.InputField(prefix='Topic you want to write: ', format=str)
    conv = dspy.InputField(prefix='Conversation history:\n', format=str)
    question = dspy.OutputField(format=str)

class AskQuestionWithPersona(dspy.Signature):
    """You are an experienced Wikipedia writer and want to edit a specific page. Besides your identity as a Wikipedia writer, you have specific focus when researching the topic.
    Now, you are chatting with an expert to get information. Ask good questions to get more useful information.
    When you have no more question to ask, say "Thank you so much for your help!" to end the conversation.
    Please only ask a question at a time and don't ask what you have asked before. Your questions should be related to the topic you want to write."""

    topic = dspy.InputField(prefix='Topic you want to write: ', format=str)
    persona = dspy.InputField(prefix='Your persona besides being a Wikipedia writer: ', format=str)
    conv = dspy.InputField(prefix='Conversation history:\n', format=str)
    question = dspy.OutputField(format=str)

class QuestionToQuery(dspy.Signature):
    """You want to answer the question using Google search. What do you type in the search box?
        Write the queries you will use in the following format:
        - query 1
        - query 2
        ...
        - query n"""

    topic = dspy.InputField(prefix='Topic you are discussing about: ', format=str)
    question = dspy.InputField(prefix='Question you want to answer: ', format=str)
    queries = dspy.OutputField(format=str)

class AnswerQuestion(dspy.Signature):
    """You are an expert who can use information effectively. You are chatting with a Wikipedia writer who wants to write a Wikipedia page on topic you know. You have gathered the related information and will now use the information to form a response.
    Make your response as informative as possible and make sure every sentence is supported by the gathered information. If [Gathered information] is not related to he [Topic] and [Question], output "Sorry, I don't have enough information to answer the question."."""

    topic = dspy.InputField(prefix='Topic you are discussing about:', format=str)
    conv = dspy.InputField(prefix='Question:\n', format=str)
    info = dspy.InputField(
        prefix='Gathered information:\n', format=str)
    answer = dspy.OutputField(
        prefix='Now give your response. (Try to use as many different sources as possible and add do not hallucinate.)\n',
        format=str
    )

class TopicExpert(dspy.Module):
    """Answer questions using search-based retrieval and answer generation. This module conducts the following steps:
    1. Generate queries from the question.
    2. Search for information using the queries.
    3. Filter out unreliable sources.
    4. Generate an answer using the retrieved information.
    """

    def __init__(self, engine: Union[dspy.dsp.LM, dspy.dsp.HFModel],
                 max_search_queries: int, search_top_k: int, retriever: Retriever):
        super().__init__()
        self.generate_queries = dspy.Predict(QuestionToQuery)
        self.retriever = retriever
        self.answer_question = dspy.Predict(AnswerQuestion)
        self.engine = engine
        self.max_search_queries = max_search_queries
        self.search_top_k = search_top_k

    def forward(self, topic: str, question: str, ground_truth_url: str):
        with dspy.settings.context(lm=self.engine):
            # Identify: Break down question into queries.
            queries = self.generate_queries(topic=topic, question=question).queries
            queries = [q.replace('-', '').strip().strip('"').strip('"').strip() for q in queries.split('\n')]
            queries = queries[:self.max_search_queries]

            # Search
            raw_results = self.retriever.retrieve(list(set(queries)), exclude_urls=[ground_truth_url])
            
            # Convert results to StormInformation objects if necessary
            searched_results = []
            for result in raw_results:
                if isinstance(result, dict):
                    searched_results.append(StormInformation.from_dict(result))
                elif isinstance(result, StormInformation):
                    searched_results.append(result)
                else:
                    logging.warning(f"Unexpected result type: {type(result)}. Skipping this result.")

            if len(searched_results) > 0:
                # Evaluate: Simplify this part by directly using the top 1 snippet.
                info = ''
                for n, r in enumerate(searched_results):
                    if isinstance(r, StormInformation) and r.snippets:
                        info += f'[{n + 1}]: {r.snippets[0]}\n\n'
                    elif isinstance(r, dict) and 'snippets' in r and r['snippets']:
                        info += f'[{n + 1}]: {r["snippets"][0]}\n\n'
                    else:
                        logging.warning(f"Result {n} does not have expected 'snippets' attribute or is empty.")

                info = ArticleTextProcessing.limit_word_count_preserve_newline(info, 1000)

                try:
                    answer = self.answer_question(topic=topic, conv=question, info=info).answer
                    answer = ArticleTextProcessing.remove_uncompleted_sentences_with_citations(answer)
                except Exception as e:
                    logging.error(f'Error occurs when generating answer: {e}')
                    answer = 'Sorry, I cannot answer this question. Please ask another question.'
            else:
                # When no information is found, the expert shouldn't hallucinate.
                answer = 'Sorry, I cannot find information for this question. Please ask another question.'

        return dspy.Prediction(queries=queries, searched_results=searched_results, answer=answer)
    
class StormKnowledgeCurationModule(KnowledgeCurationModule):
    """
    De interface voor de kenniscuratiefase. Gegeven een onderwerp, verzamelt en retourneert deze module informatie.
    """

    def __init__(self,
                 retriever: CombinedRetriever,
                 persona_generator: Optional[StormPersonaGenerator],
                 conv_simulator_lm: Union[dspy.dsp.LM, dspy.dsp.HFModel],
                 question_asker_lm: Union[dspy.dsp.LM, dspy.dsp.HFModel],
                 max_search_queries_per_turn: int,
                 search_top_k: int,
                 max_conv_turn: int,
                 max_thread_num: int,
                 num_persona: int):
        """
        Initialiseert de StormKnowledgeCurationModule met de gegeven parameters.

        Args:
            retriever: De retriever voor het ophalen van informatie.
            persona_generator: De generator voor het maken van persona's.
            conv_simulator_lm: Het taalmodel voor de conversatiesimulator.
            question_asker_lm: Het taalmodel voor het stellen van vragen.
            max_search_queries_per_turn: Maximaal aantal zoekopdrachten per beurt.
            search_top_k: Aantal top zoekresultaten om te overwegen.
            max_conv_turn: Maximaal aantal beurten in een conversatie.
            max_thread_num: Maximaal aantal threads voor parallelle uitvoering.
            num_persona: Aantal persona's om te genereren.
        """
        super().__init__(retriever)
        self.persona_generator = persona_generator
        self.conv_simulator_lm = conv_simulator_lm
        self.search_top_k = search_top_k
        self.max_thread_num = max_thread_num
        self.num_persona = num_persona
        
        self.conv_simulator = ConvSimulator(
            topic_expert_engine=conv_simulator_lm,
            question_asker_engine=question_asker_lm,
            retriever=self.retriever,
            max_search_queries_per_turn=max_search_queries_per_turn,
            search_top_k=search_top_k,
            max_turn=max_conv_turn
        )

    def _get_considered_personas(self, topic: str) -> List[str]:
        """
        Genereert een lijst van persona's voor het gegeven onderwerp.

        Args:
            topic: Het onderwerp waarvoor persona's worden gegenereerd.

        Returns:
            Een lijst van gegenereerde persona's.
        """
        return self.persona_generator.generate_persona(topic=topic, num_persona=self.num_persona)

    def _run_conversation(self, conv_simulator, topic, ground_truth_url, considered_personas,
                          callback_handler: BaseCallbackHandler) -> List[Tuple[str, List[DialogueTurn]]]:
        """
        Voert meerdere conversatiesimulaties parallel uit, elk met een verschillende persona,
        en verzamelt hun dialooggeschiedenissen.

        Args:
            conv_simulator: De functie om conversaties te simuleren.
            topic: Het onderwerp van de conversatie.
            ground_truth_url: De URL naar de grondwaarheid gerelateerd aan het gespreksonderwerp.
            considered_personas: Een lijst van persona's waarvoor de conversatiesimulaties worden uitgevoerd.
            callback_handler: Een callback-functie voor het afhandelen van gebeurtenissen tijdens de simulatie.

        Returns:
            Een lijst van tuples, waarbij elke tuple een persona en zijn bijbehorende opgeschoonde
            dialooggeschiedenis bevat.
        """
        conversations = []

        def run_conv(persona):
            return conv_simulator(
                topic=topic,
                ground_truth_url=ground_truth_url,
                persona=persona,
                callback_handler=callback_handler
            )

        max_workers = min(self.max_thread_num, len(considered_personas))

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_persona = {executor.submit(run_conv, persona): persona for persona in considered_personas}

            if streamlit_connection:
                # Zorg voor de juiste logging context bij verbinding met Streamlit frontend.
                for t in executor._threads:
                    add_script_run_ctx(t)

            for future in as_completed(future_to_persona):
                persona = future_to_persona[future]
                conv = future.result()
                conversations.append((persona, ArticleTextProcessing.clean_up_citation(conv).dlg_history))

        return conversations

    def research(self,
             topic: str,
             ground_truth_url: str,
             callback_handler: BaseCallbackHandler,
             disable_perspective: bool = False,
             return_conversation_log: bool = False) -> Union[StormInformationTable, Tuple[StormInformationTable, Dict]]:
        """
        Verzamelt informatie en kennis voor het gegeven onderwerp.

        Args:
            topic: Onderwerp van interesse in natuurlijke taal.
            ground_truth_url: URL die moet worden uitgesloten van de zoekopdracht om lekkage van de grondwaarheid te voorkomen.
            callback_handler: Handler voor callbacks tijdens het onderzoeksproces.
            disable_perspective: Als True, schakelt perspectief-geleide vraagstelling uit.
            return_conversation_log: Als True, retourneert het gespreklogboek samen met de informatietabel.

        Returns:
            collected_information: Verzamelde informatie in InformationTable type.
            conversation_log: (optioneel) Log van het conversatieproces.
        """
        if isinstance(self.retriever, CombinedRetriever):
            logging.info(f"Actieve retrievers: {', '.join(self.retriever.active_retrievers)}")

        # Identificeer persona's
        callback_handler.on_identify_perspective_start()
        if disable_perspective:
            considered_personas = [""]
        else:
            considered_personas = self._get_considered_personas(topic=topic)
        callback_handler.on_identify_perspective_end(perspectives=considered_personas)

        # Voer conversatie uit
        callback_handler.on_information_gathering_start()
        conversations = self._run_conversation(conv_simulator=self.conv_simulator,
                                            topic=topic,
                                            ground_truth_url=ground_truth_url,
                                            considered_personas=considered_personas,
                                            callback_handler=callback_handler)

        information_table = StormInformationTable(conversations)
        callback_handler.on_information_gathering_end()
        
        if return_conversation_log:
            return information_table, StormInformationTable.construct_log_dict(conversations)
        return information_table