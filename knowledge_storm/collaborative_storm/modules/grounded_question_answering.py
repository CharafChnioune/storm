# Importeer benodigde modules voor type hints en dspy functionaliteit
import dspy
from typing import Union, List

# Importeer hulpfuncties en datastructuren
from .callback import BaseCallbackHandler
from .collaborative_storm_utils import (
    trim_output_after_hint,
    format_search_results,
    extract_cited_storm_info,
    separate_citations,
)
from ...logging_wrapper import LoggingWrapper
from ...utils import ArticleTextProcessing
from ...interface import Information


class QuestionToQuery(dspy.Signature):
    """
    Definieert de structuur voor het omzetten van een vraag naar zoekquery's.
    
    Doel: Genereer effectieve zoekquery's voor het beantwoorden van een vraag of ondersteunen van een claim.
    
    Aanname: De vraag is gesteld in de context van een rondetafelgesprek over een bepaald onderwerp.
    Beperking: De output moet het gespecificeerde queryformaat volgen.
    """

    topic = dspy.InputField(prefix="Onderwerp context:", format=str)
    question = dspy.InputField(
        prefix="Ik wil informatie verzamelen over: ", format=str
    )
    queries = dspy.OutputField(prefix="Query's: \n", format=str)


class AnswerQuestion(dspy.Signature):
    """
    Definieert de structuur voor het beantwoorden van een vraag met verzamelde informatie.
    
    Doel: Genereer een informatief antwoord op basis van verzamelde gegevens.
    
    Aannames:
    - De expert kan effectief informatie gebruiken.
    - Elke zin in het antwoord moet onderbouwd zijn met verzamelde informatie.
    
    Beperkingen:
    - Bij irrelevante informatie moet het beste mogelijke antwoord gegeven worden.
    - Citaties moeten inline gebruikt worden met [1], [2], etc.
    - Geen aparte bronnenlijst nodig.
    - Schrijfstijl moet formeel zijn, tenzij anders aangegeven.
    """

    topic = dspy.InputField(prefix="Onderwerp waarover u discussieert:", format=str)
    question = dspy.InputField(prefix="U wilt inzicht geven over: ", format=str)
    info = dspy.InputField(prefix="Verzamelde informatie:\n", format=str)
    style = dspy.InputField(prefix="Stijl van uw antwoord moet zijn:", format=str)
    answer = dspy.OutputField(
        prefix="Geef nu uw antwoord. (Probeer zoveel mogelijk verschillende bronnen te gebruiken en niet te hallucineren.)",
        format=str,
    )


class AnswerQuestionModule(dspy.Module):
    def __init__(
        self,
        retriever: dspy.Retrieve,
        max_search_queries: int,
        question_answering_lm: Union[dspy.dsp.LM, dspy.dsp.HFModel],
        logging_wrapper: LoggingWrapper,
    ):
        """
        Initialiseert de module voor het beantwoorden van vragen.
        
        :param retriever: Module voor het ophalen van informatie
        :param max_search_queries: Maximaal aantal zoekquery's
        :param question_answering_lm: Taalmodel voor vraagbeantwoording
        :param logging_wrapper: Wrapper voor logging
        """
        super().__init__()
        self.question_answering_lm = question_answering_lm
        self.question_to_query = dspy.Predict(QuestionToQuery)
        self.answer_question = dspy.Predict(AnswerQuestion)
        self.retriever = retriever
        self.max_search_queries = max_search_queries
        self.logging_wrapper = logging_wrapper

    def retrieve_information(self, topic, question):
        """
        Haalt relevante informatie op voor een gegeven vraag.
        
        :param topic: Het onderwerp van de vraag
        :param question: De te beantwoorden vraag
        :return: Een tuple met query's en zoekresultaten
        """
        # Genereer zoekquery's op basis van de vraag
        with self.logging_wrapper.log_event(
            f"AnswerQuestionModule.question_to_query ({hash(question)})"
        ):
            with dspy.settings.context(lm=self.question_answering_lm):
                queries = self.question_to_query(topic=topic, question=question).queries
            queries = trim_output_after_hint(queries, hint="Query's:")
            queries = [
                q.replace("-", "").strip().strip('"').strip('"').strip()
                for q in queries.split("\n")
            ]
            queries = queries[: self.max_search_queries]
        self.logging_wrapper.add_query_count(count=len(queries))

        # Haal informatie op met de gegenereerde query's
        with self.logging_wrapper.log_event(
            f"AnswerQuestionModule.retriever.retrieve ({hash(question)})"
        ):
            searched_results: List[Information] = self.retriever.retrieve(
                list(set(queries)), exclude_urls=[]
            )
        # Voeg de vraag toe aan de metadata van de zoekresultaten
        for storm_info in searched_results:
            storm_info.meta["question"] = question
        return queries, searched_results

    def forward(
        self,
        topic: str,
        question: str,
        mode: str = "brief",
        style: str = "conversational",
        callback_handler: BaseCallbackHandler = None,
    ):
        """
        Verwerkt een onderwerp en vraag om een antwoord te genereren met relevante informatie en citaties.

        :param topic: Het onderwerp van interesse
        :param question: De specifieke vraag gerelateerd aan het onderwerp
        :param mode: Modus van samenvatting ('brief' of 'extensive')
        :param style: Gewenste stijl van het antwoord
        :param callback_handler: Optionele callback handler voor voortgangsrapportage
        :return: dspy.Prediction object met gegenereerd antwoord en metadata
        """
        # Haal relevante informatie op
        if callback_handler is not None:
            callback_handler.on_expert_information_collection_start()
        queries, searched_results = self.retrieve_information(
            topic=topic, question=question
        )
        if callback_handler is not None:
            callback_handler.on_expert_information_collection_end(searched_results)

        # Formatteer de opgehaalde informatie voor antwoordgeneratie
        info_text, index_to_information_mapping = format_search_results(
            searched_results, mode=mode
        )
        answer = "Sorry, er is onvoldoende informatie om de vraag te beantwoorden."

        # Genereer een antwoord als er informatie beschikbaar is
        if info_text:
            with self.logging_wrapper.log_event(
                f"AnswerQuestionModule.answer_question ({hash(question)})"
            ):
                with dspy.settings.context(
                    lm=self.question_answering_lm, show_guidelines=False
                ):
                    answer = self.answer_question(
                        topic=topic, question=question, info=info_text, style=style
                    ).answer
                    answer = ArticleTextProcessing.remove_uncompleted_sentences_with_citations(
                        answer
                    )
                    answer = trim_output_after_hint(
                        answer,
                        hint="Geef nu uw antwoord. (Probeer zoveel mogelijk verschillende bronnen te gebruiken en niet te hallucineren.)",
                    )
                    # Zorg voor consistente citatienotatie: [1, 2] -> [1][2]
                    answer = separate_citations(answer)
                    if callback_handler is not None:
                        callback_handler.on_expert_utterance_generation_end()

        # Extraheer geciteerde informatie uit het antwoord
        cited_searched_results = extract_cited_storm_info(
            response=answer, index_to_storm_info=index_to_information_mapping
        )

        return dspy.Prediction(
            question=question,
            queries=queries,
            raw_retrieved_info=searched_results,
            cited_info=cited_searched_results,
            response=answer,
        )