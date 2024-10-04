import json
import logging
import os
from dataclasses import dataclass, field
from typing import Union, Literal, Optional

import dspy

from .modules.article_generation import StormArticleGenerationModule
from .modules.article_polish import StormArticlePolishingModule
from .modules.callback import BaseCallbackHandler
from .modules.knowledge_curation import StormKnowledgeCurationModule
from .modules.outline_generation import StormOutlineGenerationModule
from .modules.persona_generator import StormPersonaGenerator
from .modules.storm_dataclass import StormInformationTable, StormArticle
from ..interface import Engine, LMConfigs, Retriever
from ..lm import OpenAIModel, AzureOpenAIModel, OllamaClient
from ..utils import FileIOHelper, makeStringRed, truncate_filename


class STORMWikiLMConfigs(LMConfigs):
    """Configuraties voor LLM gebruikt in verschillende delen van STORM.

    Aangezien verschillende onderdelen in het STORM-framework verschillende complexiteit hebben, 
    gebruiken we verschillende LLM-configuraties om een balans te bereiken tussen kwaliteit en efficiëntie. 
    Als er geen specifieke configuratie wordt opgegeven, gebruiken we de standaardopstelling uit het paper.
    """

    def __init__(self):
        # LLM gebruikt in conversatiesimulator, behalve voor het stellen van vragen
        self.conv_simulator_lm = None
        # LLM gebruikt voor het stellen van vragen
        self.question_asker_lm = None  
        # LLM gebruikt voor het genereren van outlines
        self.outline_gen_lm = None
        # LLM gebruikt voor het genereren van artikelen
        self.article_gen_lm = None
        # LLM gebruikt voor het polijsten van artikelen
        self.article_polish_lm = None

    def init_openai_model(
        self,
        openai_api_key: str,
        azure_api_key: str,
        openai_type: Literal["openai", "azure"],
        api_base: Optional[str] = None,
        api_version: Optional[str] = None,
        temperature: Optional[float] = 1.0,
        top_p: Optional[float] = 0.9,
    ):
        """Legacy: Overeenkomend met de originele setup in het NAACL'24 paper."""
        # Configuratie voor Azure OpenAI
        azure_kwargs = {
            "api_key": azure_api_key,
            "temperature": temperature,
            "top_p": top_p,
            "api_base": api_base,
            "api_version": api_version,
        }

        # Configuratie voor OpenAI
        openai_kwargs = {
            "api_key": openai_api_key,
            "api_provider": "openai",
            "temperature": temperature,
            "top_p": top_p,
            "api_base": None,
        }
        
        # Initialiseer modellen op basis van het opgegeven OpenAI-type
        if openai_type and openai_type == "openai":
            # Configuratie voor OpenAI modellen
            self.conv_simulator_lm = OpenAIModel(
                model="gpt-4o-mini-2024-07-18", max_tokens=500, **openai_kwargs
            )
            self.question_asker_lm = OpenAIModel(
                model="gpt-4o-mini-2024-07-18", max_tokens=500, **openai_kwargs
            )
            # 1/12/2024: Update gpt-4 naar gpt-4-1106-preview. (Momenteel behouden we de originele setup bij gebruik van Azure.)
            self.outline_gen_lm = OpenAIModel(
                model="gpt-4-0125-preview", max_tokens=400, **openai_kwargs
            )
            self.article_gen_lm = OpenAIModel(
                model="gpt-4o-2024-05-13", max_tokens=700, **openai_kwargs
            )
            self.article_polish_lm = OpenAIModel(
                model="gpt-4o-2024-05-13", max_tokens=4000, **openai_kwargs
            )
        elif openai_type and openai_type == "azure":
            # Configuratie voor Azure OpenAI modellen
            self.conv_simulator_lm = OpenAIModel(
                model="gpt-4o-mini-2024-07-18", max_tokens=500, **openai_kwargs
            )
            self.question_asker_lm = AzureOpenAIModel(
                model="gpt-4o-mini-2024-07-18",
                max_tokens=500,
                **azure_kwargs,
                model_type="chat",
            )
            # Gebruik combinatie van OpenAI en Azure OpenAI omdat Azure OpenAI geen gpt-4 ondersteunt in standaard implementatie
            self.outline_gen_lm = AzureOpenAIModel(
                model="gpt-4o", max_tokens=400, **azure_kwargs, model_type="chat"
            )
            self.article_gen_lm = AzureOpenAIModel(
                model="gpt-4o-mini-2024-07-18",
                max_tokens=700,
                **azure_kwargs,
                model_type="chat",
            )
            self.article_polish_lm = AzureOpenAIModel(
                model="gpt-4o-mini-2024-07-18",
                max_tokens=4000,
                **azure_kwargs,
                model_type="chat",
            )
        else:
            logging.warning(
                "Geen geldige OpenAI API-provider opgegeven. Kan standaard LLM-configuraties niet gebruiken."
            )

    def init_ollama_model(
        self,
        model: str,
        port: int,
        url: str = "http://localhost",
        temperature: float = 1.0,
        top_p: float = 0.9,
        max_tokens: int = 500
    ):
        """Initialiseer Ollama modellen voor verschillende onderdelen van STORM."""
        ollama_kwargs = {
            "model": model,
            "port": port,
            "url": url,
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens
        }

        self.conv_simulator_lm = OllamaClient(**ollama_kwargs)
        self.question_asker_lm = OllamaClient(**ollama_kwargs)
        self.outline_gen_lm = OllamaClient(**ollama_kwargs)
        self.article_gen_lm = OllamaClient(**ollama_kwargs)
        self.article_polish_lm = OllamaClient(**ollama_kwargs)

    # Methoden om specifieke LLM's in te stellen
    def set_conv_simulator_lm(self, model: Union[dspy.dsp.LM, dspy.dsp.HFModel]):
        self.conv_simulator_lm = model

    def set_question_asker_lm(self, model: Union[dspy.dsp.LM, dspy.dsp.HFModel]):
        self.question_asker_lm = model

    def set_outline_gen_lm(self, model: Union[dspy.dsp.LM, dspy.dsp.HFModel]):
        self.outline_gen_lm = model

    def set_article_gen_lm(self, model: Union[dspy.dsp.LM, dspy.dsp.HFModel]):
        self.article_gen_lm = model

    def set_article_polish_lm(self, model: Union[dspy.dsp.LM, dspy.dsp.HFModel]):
        self.article_polish_lm = model


@dataclass
class STORMWikiRunnerArguments:
    """Argumenten voor het beheren van de STORM Wiki pipeline."""

    output_dir: str = field(
        metadata={"help": "Uitvoermap voor de resultaten."},
    )
    max_conv_turn: int = field(
        default=3,
        metadata={
            "help": "Maximaal aantal vragen in conversationeel vragen stellen."
        },
    )
    max_perspective: int = field(
        default=3,
        metadata={
            "help": "Maximaal aantal perspectieven om te overwegen bij perspectief-gestuurd vragen stellen."
        },
    )
    max_search_queries_per_turn: int = field(
        default=3,
        metadata={"help": "Maximaal aantal zoekopdrachten om te overwegen in elke beurt."},
    )
    disable_perspective: bool = field(
        default=False,
        metadata={"help": "Indien True, schakel perspectief-gestuurd vragen stellen uit."},
    )
    search_top_k: int = field(
        default=3,
        metadata={"help": "Top k zoekresultaten om te overwegen voor elke zoekopdracht."},
    )
    retrieve_top_k: int = field(
        default=3,
        metadata={"help": "Top k verzamelde referenties voor elke sectietitel."},
    )
    max_thread_num: int = field(
        default=10,
        metadata={
            "help": "Maximaal aantal threads om te gebruiken. "
            "Overweeg dit te verlagen als je blijft 'Exceed rate limit' fouten krijgt bij het aanroepen van de LM API."
        },
    )


class STORMWikiRunner(Engine):
    """STORM Wiki pipeline runner."""

    def __init__(
        self, args: STORMWikiRunnerArguments, lm_configs: STORMWikiLMConfigs, rm
    ):
        super().__init__(lm_configs=lm_configs)
        self.args = args
        self.lm_configs = lm_configs

        # Initialiseer componenten voor de STORM pipeline
        self.retriever = Retriever(rm=rm, max_thread=self.args.max_thread_num)
        storm_persona_generator = StormPersonaGenerator(
            self.lm_configs.question_asker_lm
        )
        self.storm_knowledge_curation_module = StormKnowledgeCurationModule(
            retriever=self.retriever,
            persona_generator=storm_persona_generator,
            conv_simulator_lm=self.lm_configs.conv_simulator_lm,
            question_asker_lm=self.lm_configs.question_asker_lm,
            max_search_queries_per_turn=self.args.max_search_queries_per_turn,
            search_top_k=self.args.search_top_k,
            max_conv_turn=self.args.max_conv_turn,
            max_thread_num=self.args.max_thread_num,
        )
        self.storm_outline_generation_module = StormOutlineGenerationModule(
            outline_gen_lm=self.lm_configs.outline_gen_lm
        )
        self.storm_article_generation = StormArticleGenerationModule(
            article_gen_lm=self.lm_configs.article_gen_lm,
            retrieve_top_k=self.args.retrieve_top_k,
            max_thread_num=self.args.max_thread_num,
        )
        self.storm_article_polishing_module = StormArticlePolishingModule(
            article_gen_lm=self.lm_configs.article_gen_lm,
            article_polish_lm=self.lm_configs.article_polish_lm,
        )

        self.lm_configs.init_check()
        self.apply_decorators()

    def run_knowledge_curation_module(
        self,
        ground_truth_url: str = "None",
        callback_handler: BaseCallbackHandler = None,
    ) -> StormInformationTable:
        """
        Voert de kenniscuratie module uit.
        
        Genereert een informatietabel en conversatielog door onderzoek te doen naar het opgegeven onderwerp.
        Slaat de resultaten op in JSON-bestanden.
        """
        information_table, conversation_log = (
            self.storm_knowledge_curation_module.research(
                topic=self.topic,
                ground_truth_url=ground_truth_url,
                callback_handler=callback_handler,
                max_perspective=self.args.max_perspective,
                disable_perspective=False,
                return_conversation_log=True,
            )
        )

        FileIOHelper.dump_json(
            conversation_log,
            os.path.join(self.article_output_dir, "conversation_log.json"),
        )
        information_table.dump_url_to_info(
            os.path.join(self.article_output_dir, "raw_search_results.json")
        )
        return information_table

    def run_outline_generation_module(
        self,
        information_table: StormInformationTable,
        callback_handler: BaseCallbackHandler = None,
    ) -> StormArticle:
        """
        Voert de outline generatie module uit.
        
        Genereert een outline en een concept outline op basis van de informatietabel.
        Slaat de resultaten op in tekstbestanden.
        """
        outline, draft_outline = self.storm_outline_generation_module.generate_outline(
            topic=self.topic,
            information_table=information_table,
            return_draft_outline=True,
            callback_handler=callback_handler,
        )
        outline.dump_outline_to_file(
            os.path.join(self.article_output_dir, "storm_gen_outline.txt")
        )
        draft_outline.dump_outline_to_file(
            os.path.join(self.article_output_dir, "direct_gen_outline.txt")
        )
        return outline

    def run_article_generation_module(
        self,
        outline: StormArticle,
        information_table=StormInformationTable,
        callback_handler: BaseCallbackHandler = None,
    ) -> StormArticle:
        """
        Voert de artikel generatie module uit.
        
        Genereert een concept artikel op basis van de outline en informatietabel.
        Slaat het resultaat op in een tekstbestand en de referenties in een JSON-bestand.
        """
        draft_article = self.storm_article_generation.generate_article(
            topic=self.topic,
            information_table=information_table,
            article_with_outline=outline,
            callback_handler=callback_handler,
        )
        draft_article.dump_article_as_plain_text(
            os.path.join(self.article_output_dir, "storm_gen_article.txt")
        )
        draft_article.dump_reference_to_file(
            os.path.join(self.article_output_dir, "url_to_info.json")
        )
        return draft_article

    def run_article_polishing_module(
        self, draft_article: StormArticle, remove_duplicate: bool = False
    ) -> StormArticle:
        """
        Voert de artikel polijst module uit.
        
        Polijst het concept artikel en verwijdert optioneel dubbele inhoud.
        Slaat het resultaat op in een tekstbestand.
        """
        polished_article = self.storm_article_polishing_module.polish_article(
            topic=self.topic,
            draft_article=draft_article,
            remove_duplicate=remove_duplicate,
        )
        FileIOHelper.write_str(
            polished_article.to_string(),
            os.path.join(self.article_output_dir, "storm_gen_article_polished.txt"),
        )
        return polished_article

    def post_run(self):
        """
        Voert post-run operaties uit, waaronder:
        1. Het dumpen van de run configuratie.
        2. Het dumpen van de LLM aanroepgeschiedenis.
        """
        config_log = self.lm_configs.log()
        FileIOHelper.dump_json(
            config_log, os.path.join(self.article_output_dir, "run_config.json")
        )

        llm_call_history = self.lm_configs.collect_and_reset_lm_history()
        with open(
            os.path.join(self.article_output_dir, "llm_call_history.jsonl"), "w"
        ) as f:
            for call in llm_call_history:
                if "kwargs" in call:
                    call.pop(
                        "kwargs"
                    )  # Alle kwargs worden samen gedumpt naar run_config.json.
                f.write(json.dumps(call) + "\n")

    def _load_information_table_from_local_fs(self, information_table_local_path):
        """Laadt de informatietabel van het lokale bestandssysteem."""
        assert os.path.exists(information_table_local_path), makeStringRed(
            f"{information_table_local_path} bestaat niet. Stel het --do-research argument in om de conversation_log.json voor dit onderwerp voor te bereiden."
        )
        return StormInformationTable.from_conversation_log_file(
            information_table_local_path
        )

    def _load_outline_from_local_fs(self, topic, outline_local_path):
        """Laadt de outline van het lokale bestandssysteem."""
        assert os.path.exists(outline_local_path), makeStringRed(
            f"{outline_local_path} bestaat niet. Stel het --do-generate-outline argument in om de storm_gen_outline.txt voor dit onderwerp voor te bereiden."
        )
        return StormArticle.from_outline_file(topic=topic, file_path=outline_local_path)

    def _load_draft_article_from_local_fs(
        self, topic, draft_article_path, url_to_info_path
    ):
        """Laadt het concept artikel van het lokale bestandssysteem."""
        assert os.path.exists(draft_article_path), makeStringRed(
            f"{draft_article_path} bestaat niet. Stel het --do-generate-article argument in om de storm_gen_article.txt voor dit onderwerp voor te bereiden."
        )
        assert os.path.exists(url_to_info_path), makeStringRed(
            f"{url_to_info_path} bestaat niet. Stel het --do-generate-article argument in om de url_to_info.json voor dit onderwerp voor te bereiden."
        )
        article_text = FileIOHelper.load_str(draft_article_path)
        references = FileIOHelper.load_json(url_to_info_path)
        return StormArticle.from_string(
            topic_name=topic, article_text=article_text, references=references
        )

    def run(
        self,
        topic: str,
        ground_truth_url: str = "",
        do_research: bool = True,
        do_generate_outline: bool = True,
        do_generate_article: bool = True,
        do_polish_article: bool = True,
        remove_duplicate: bool = False,
        callback_handler: BaseCallbackHandler = BaseCallbackHandler(),
    ):
        """
        Voert de STORM pipeline uit.

        Args:
            topic: Het te onderzoeken onderwerp.
            ground_truth_url: Een ground truth URL met een gecureerd artikel over het onderwerp. De URL wordt uitgesloten.
            do_research: Indien True, onderzoek het onderwerp via informatiezoekende conversatie;
             indien False, verwacht dat conversation_log.json en raw_search_results.json bestaan in de uitvoermap.
            do_generate_outline: Indien True, genereer een outline voor het onderwerp;
             indien False, verwacht dat storm_gen_outline.txt bestaat in de uitvoermap.
            do_generate_article: Indien True, genereer een gecureerd artikel voor het onderwerp;
             indien False, verwacht dat storm_gen_article.txt bestaat in de uitvoermap.
            do_polish_article: Indien True, polijst het artikel door een samenvattingssectie toe te voegen en (optioneel)
             dubbele inhoud te verwijderen.
            remove_duplicate: Indien True, verwijder dubbele inhoud.
            callback_handler: Een callback handler om de tussenresultaten te verwerken.
        """
        assert (
            do_research
            or do_generate_outline
            or do_generate_article
            or do_polish_article
        ), makeStringRed(
            "Geen actie gespecificeerd. Stel ten minste een van --do-research, --do-generate-outline, --do-generate-article, --do-polish-article in"
        )

        self.topic = topic
        self.article_dir_name = truncate_filename(
            topic.replace(" ", "_").replace("/", "_")
        )
        self.article_output_dir = os.path.join(
            self.args.output_dir, self.article_dir_name
        )
        os.makedirs(self.article_output_dir, exist_ok=True)

        # Onderzoeksmodule
        information_table: StormInformationTable = None
        if do_research:
            information_table = self.run_knowledge_curation_module(
                ground_truth_url=ground_truth_url, callback_handler=callback_handler
            )
        
        # Outline generatiemodule
        outline: StormArticle = None
        if do_generate_outline:
            # Laad informatietabel als deze niet is geïnitialiseerd
            if information_table is None:
                information_table = self._load_information_table_from_local_fs(
                    os.path.join(self.article_output_dir, "conversation_log.json")
                )
            outline = self.run_outline_generation_module(
                information_table=information_table, callback_handler=callback_handler
            )

        # Artikel generatiemodule
        draft_article: StormArticle = None
        if do_generate_article:
            if information_table is None:
                information_table = self._load_information_table_from_local_fs(
                    os.path.join(self.article_output_dir, "conversation_log.json")
                )
            if outline is None:
                outline = self._load_outline_from_local_fs(
                    topic=topic,
                    outline_local_path=os.path.join(
                        self.article_output_dir, "storm_gen_outline.txt"
                    ),
                )
            draft_article = self.run_article_generation_module(
                outline=outline,
                information_table=information_table,
                callback_handler=callback_handler,
            )

        # Artikel polijstmodule
        if do_polish_article:
            if draft_article is None:
                draft_article_path = os.path.join(
                    self.article_output_dir, "storm_gen_article.txt"
                )
                url_to_info_path = os.path.join(
                    self.article_output_dir, "url_to_info.json"
                )
                draft_article = self._load_draft_article_from_local_fs(
                    topic=topic,
                    draft_article_path=draft_article_path,
                    url_to_info_path=url_to_info_path,
                )
            self.run_article_polishing_module(
                draft_article=draft_article, remove_duplicate=remove_duplicate
            )