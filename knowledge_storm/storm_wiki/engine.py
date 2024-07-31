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
from .modules.retriever import StormRetriever
from .modules.storm_dataclass import StormInformationTable, StormArticle
from ..interface import Engine, LMConfigs
from ..lm import OpenAIModel
from ..utils import FileIOHelper, makeStringRed

# Configureer logging
logging.basicConfig(level=logging.DEBUG, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.FileHandler("storm_wiki_runner.log"),
                        logging.StreamHandler()
                    ])

logger = logging.getLogger(__name__)

class STORMWikiLMConfigs(LMConfigs):
    """Configurations for LLM used in different parts of STORM."""

    def __init__(self):
        super().__init__()
        logger.info("Initialiseren STORMWikiLMConfigs")
        self.conv_simulator_lm = None  # LLM used in conversation simulator except for question asking.
        self.question_asker_lm = None  # LLM used in question asking.
        self.outline_gen_lm = None  # LLM used in outline generation.
        self.article_gen_lm = None  # LLM used in article generation.
        self.article_polish_lm = None  # LLM used in article polishing.

    def init_openai_model(
            self,
            openai_api_key: str,
            openai_type: Literal["openai", "azure"],
            api_base: Optional[str] = None,
            api_version: Optional[str] = None,
            temperature: Optional[float] = 1.0,
            top_p: Optional[float] = 0.9
    ):
        """Legacy: Corresponding to the original setup in the NAACL'24 paper."""
        logger.info(f"Initialiseren OpenAI model: type={openai_type}, temperature={temperature}, top_p={top_p}")
        openai_kwargs = {
            'api_key': openai_api_key,
            'api_provider': openai_type,
            'temperature': temperature,
            'top_p': top_p,
            'api_base': None
        }
        if openai_type and openai_type == 'openai':
            logger.debug("Configureren OpenAI modellen")
            self.conv_simulator_lm = OpenAIModel(model='gpt-3.5-turbo-instruct',
                                                 max_tokens=500, **openai_kwargs)
            self.question_asker_lm = OpenAIModel(model='gpt-3.5-turbo',
                                                 max_tokens=500, **openai_kwargs)
            self.outline_gen_lm = OpenAIModel(model='gpt-4-0125-preview',
                                              max_tokens=400, **openai_kwargs)
            self.article_gen_lm = OpenAIModel(model='gpt-4o-2024-05-13',
                                              max_tokens=700, **openai_kwargs)
            self.article_polish_lm = OpenAIModel(model='gpt-4o-2024-05-13',
                                                 max_tokens=4000, **openai_kwargs)
        else:
            logger.warning('Geen geldige OpenAI API provider opgegeven. Kan standaard LLM-configuraties niet gebruiken.')

    def set_conv_simulator_lm(self, model: Union[dspy.dsp.LM, dspy.dsp.HFModel]):
        logger.info(f"Instellen conv_simulator_lm: {type(model)}")
        self.conv_simulator_lm = model

    def set_question_asker_lm(self, model: Union[dspy.dsp.LM, dspy.dsp.HFModel]):
        logger.info(f"Instellen question_asker_lm: {type(model)}")
        self.question_asker_lm = model

    def set_outline_gen_lm(self, model: Union[dspy.dsp.LM, dspy.dsp.HFModel]):
        logger.info(f"Instellen outline_gen_lm: {type(model)}")
        self.outline_gen_lm = model

    def set_article_gen_lm(self, model: Union[dspy.dsp.LM, dspy.dsp.HFModel]):
        logger.info(f"Instellen article_gen_lm: {type(model)}")
        self.article_gen_lm = model

    def set_article_polish_lm(self, model: Union[dspy.dsp.LM, dspy.dsp.HFModel]):
        logger.info(f"Instellen article_polish_lm: {type(model)}")
        self.article_polish_lm = model


@dataclass
class STORMWikiRunnerArguments:
    """Arguments for controlling the STORM Wiki pipeline."""
    output_dir: str = field(
        metadata={"help": "Output directory for the results."},
    )
    max_conv_turn: int = field(
        default=3,
        metadata={"help": "Maximum number of questions in conversational question asking."},
    )
    max_perspective: int = field(
        default=3,
        metadata={"help": "Maximum number of perspectives to consider in perspective-guided question asking."},
    )
    max_search_queries_per_turn: int = field(
        default=3,
        metadata={"help": "Maximum number of search queries to consider in each turn."},
    )
    disable_perspective: bool = field(
        default=False,
        metadata={"help": "If True, disable perspective-guided question asking."},
    )
    search_top_k: int = field(
        default=3,
        metadata={"help": "Top k search results to consider for each search query."},
    )
    retrieve_top_k: int = field(
        default=3,
        metadata={"help": "Top k collected references for each section title."},
    )
    max_thread_num: int = field(
        default=10,
        metadata={"help": "Maximum number of threads to use."},
    )


class STORMWikiRunner(Engine):
    """STORM Wiki pipeline runner."""

    def __init__(self,
                 args: STORMWikiRunnerArguments,
                 lm_configs: STORMWikiLMConfigs,
                 rm):
        logger.info("Initialiseren STORMWikiRunner")
        super().__init__(lm_configs=lm_configs)
        self.args = args
        self.lm_configs = lm_configs

        logger.debug(f"Initialiseren StormRetriever met k={self.args.retrieve_top_k}")
        self.retriever = StormRetriever(rm=rm, k=self.args.retrieve_top_k)
        
        logger.debug("Initialiseren StormPersonaGenerator")
        storm_persona_generator = StormPersonaGenerator(self.lm_configs.question_asker_lm)
        
        logger.debug("Initialiseren StormKnowledgeCurationModule")
        self.storm_knowledge_curation_module = StormKnowledgeCurationModule(
            retriever=self.retriever,
            persona_generator=storm_persona_generator,
            conv_simulator_lm=self.lm_configs.conv_simulator_lm,
            question_asker_lm=self.lm_configs.question_asker_lm,
            max_search_queries_per_turn=self.args.max_search_queries_per_turn,
            search_top_k=self.args.search_top_k,
            max_conv_turn=self.args.max_conv_turn,
            max_thread_num=self.args.max_thread_num
        )
        
        logger.debug("Initialiseren StormOutlineGenerationModule")
        self.storm_outline_generation_module = StormOutlineGenerationModule(
            outline_gen_lm=self.lm_configs.outline_gen_lm
        )
        
        logger.debug("Initialiseren StormArticleGenerationModule")
        self.storm_article_generation = StormArticleGenerationModule(
            article_gen_lm=self.lm_configs.article_gen_lm,
            retrieve_top_k=self.args.retrieve_top_k,
            max_thread_num=self.args.max_thread_num
        )
        
        logger.debug("Initialiseren StormArticlePolishingModule")
        self.storm_article_polishing_module = StormArticlePolishingModule(
            article_gen_lm=self.lm_configs.article_gen_lm,
            article_polish_lm=self.lm_configs.article_polish_lm
        )

        logger.debug("Uitvoeren init_check en apply_decorators")
        self.lm_configs.init_check()
        self.apply_decorators()

    def run_knowledge_curation_module(self,
                                      ground_truth_url: str = "None",
                                      callback_handler: BaseCallbackHandler = None) -> StormInformationTable:
        logger.info(f"Starten knowledge curation module voor topic: {self.topic}")
        information_table, conversation_log = self.storm_knowledge_curation_module.research(
            topic=self.topic,
            ground_truth_url=ground_truth_url,
            callback_handler=callback_handler,
            max_perspective=self.args.max_perspective,
            disable_perspective=False,
            return_conversation_log=True
        )
        logger.debug(f"Knowledge curation voltooid. Aantal conversations: {len(conversation_log)}")

        conversation_log_path = os.path.join(self.article_output_dir, 'conversation_log.json')
        FileIOHelper.dump_json(conversation_log, conversation_log_path)
        logger.debug(f"Conversation log opgeslagen in: {conversation_log_path}")

        raw_search_results_path = os.path.join(self.article_output_dir, 'raw_search_results.json')
        information_table.dump_url_to_info(raw_search_results_path)
        logger.debug(f"Raw search results opgeslagen in: {raw_search_results_path}")

        return information_table

    def run_outline_generation_module(self,
                                      information_table: StormInformationTable,
                                      callback_handler: BaseCallbackHandler = None) -> StormArticle:
        logger.info("Starten outline generation module")
        outline, draft_outline = self.storm_outline_generation_module.generate_outline(
            topic=self.topic,
            information_table=information_table,
            return_draft_outline=True,
            callback_handler=callback_handler
        )
        logger.debug("Outline generatie voltooid")

        outline_path = os.path.join(self.article_output_dir, 'storm_gen_outline.txt')
        outline.dump_outline_to_file(outline_path)
        logger.debug(f"Outline opgeslagen in: {outline_path}")

        draft_outline_path = os.path.join(self.article_output_dir, "direct_gen_outline.txt")
        draft_outline.dump_outline_to_file(draft_outline_path)
        logger.debug(f"Draft outline opgeslagen in: {draft_outline_path}")

        return outline

    def run_article_generation_module(self,
                                      outline: StormArticle,
                                      information_table=StormInformationTable,
                                      callback_handler: BaseCallbackHandler = None) -> StormArticle:
        logger.info("Starten article generation module")
        draft_article = self.storm_article_generation.generate_article(
            topic=self.topic,
            information_table=information_table,
            article_with_outline=outline,
            callback_handler=callback_handler
        )
        logger.debug("Article generatie voltooid")

        draft_article_path = os.path.join(self.article_output_dir, 'storm_gen_article.txt')
        draft_article.dump_article_as_plain_text(draft_article_path)
        logger.debug(f"Draft article opgeslagen in: {draft_article_path}")

        reference_path = os.path.join(self.article_output_dir, 'url_to_info.json')
        draft_article.dump_reference_to_file(reference_path)
        logger.debug(f"Referenties opgeslagen in: {reference_path}")

        return draft_article

    def run_article_polishing_module(self,
                                     draft_article: StormArticle,
                                     remove_duplicate: bool = False) -> StormArticle:
        logger.info(f"Starten article polishing module (remove_duplicate={remove_duplicate})")
        polished_article = self.storm_article_polishing_module.polish_article(
            topic=self.topic,
            draft_article=draft_article,
            remove_duplicate=remove_duplicate
        )
        logger.debug("Article polishing voltooid")

        polished_article_path = os.path.join(self.article_output_dir, 'storm_gen_article_polished.txt')
        FileIOHelper.write_str(polished_article.to_string(), polished_article_path)
        logger.debug(f"Gepolijst artikel opgeslagen in: {polished_article_path}")

        return polished_article

    def post_run(self):
        """
        Post-run operations, including:
        1. Dumping the run configuration.
        2. Dumping the LLM call history.
        """
        logger.info("Uitvoeren post-run operaties")
        config_log = self.lm_configs.log()
        config_log_path = os.path.join(self.article_output_dir, 'run_config.json')
        FileIOHelper.dump_json(config_log, config_log_path)
        logger.debug(f"Run configuratie opgeslagen in: {config_log_path}")

        llm_call_history = self.lm_configs.collect_and_reset_lm_history()
        llm_history_path = os.path.join(self.article_output_dir, 'llm_call_history.jsonl')
        with open(llm_history_path, 'w') as f:
            for call in llm_call_history:
                if 'kwargs' in call:
                    call.pop('kwargs')  # All kwargs are dumped together to run_config.json.
                f.write(json.dumps(call) + '\n')
        logger.debug(f"LLM call geschiedenis opgeslagen in: {llm_history_path}")

    def _load_information_table_from_local_fs(self, information_table_local_path):
        logger.info(f"Laden information table van: {information_table_local_path}")
        assert os.path.exists(information_table_local_path), makeStringRed(
            f"{information_table_local_path} bestaat niet. Stel --do-research in om de conversation_log.json voor dit onderwerp voor te bereiden.")
        return StormInformationTable.from_conversation_log_file(information_table_local_path)

    def _load_outline_from_local_fs(self, topic, outline_local_path):
            logger.info(f"Laden outline van: {outline_local_path}")
            assert os.path.exists(outline_local_path), makeStringRed(
                f"{outline_local_path} bestaat niet. Stel --do-generate-outline in om de storm_gen_outline.txt voor dit onderwerp voor te bereiden.")
            outline = StormArticle.from_outline_file(topic=topic, file_path=outline_local_path)
            logger.debug(f"Outline geladen voor topic: {topic}")
            return outline

    def _load_draft_article_from_local_fs(self, topic, draft_article_path, url_to_info_path):
        logger.info(f"Laden draft article van: {draft_article_path}")
        logger.info(f"Laden url_to_info van: {url_to_info_path}")
        assert os.path.exists(draft_article_path), makeStringRed(
            f"{draft_article_path} bestaat niet. Stel --do-generate-article in om de storm_gen_article.txt voor dit onderwerp voor te bereiden.")
        assert os.path.exists(url_to_info_path), makeStringRed(
            f"{url_to_info_path} bestaat niet. Stel --do-generate-article in om de url_to_info.json voor dit onderwerp voor te bereiden.")
        article_text = FileIOHelper.load_str(draft_article_path)
        references = FileIOHelper.load_json(url_to_info_path)
        draft_article = StormArticle.from_string(topic_name=topic, article_text=article_text, references=references)
        logger.debug(f"Draft article en referenties geladen voor topic: {topic}")
        return draft_article

    def run(self,
            topic: str,
            ground_truth_url: str = '',
            do_research: bool = True,
            do_generate_outline: bool = True,
            do_generate_article: bool = True,
            do_polish_article: bool = True,
            remove_duplicate: bool = False,
            callback_handler: BaseCallbackHandler = BaseCallbackHandler()):
        """
        Run the STORM pipeline.
        """
        logger.info(f"Starten STORM pipeline voor topic: {topic}")
        logger.debug(f"Parameters: do_research={do_research}, do_generate_outline={do_generate_outline}, "
                     f"do_generate_article={do_generate_article}, do_polish_article={do_polish_article}, "
                     f"remove_duplicate={remove_duplicate}, ground_truth_url={ground_truth_url}")

        assert do_research or do_generate_outline or do_generate_article or do_polish_article, \
            makeStringRed(
                "Geen actie gespecificeerd. Stel ten minste een van --do-research, --do-generate-outline, --do-generate-article, --do-polish-article in")

        self.topic = topic
        self.article_dir_name = topic.replace(' ', '_').replace('/', '_')
        self.article_output_dir = os.path.join(self.args.output_dir, self.article_dir_name)
        os.makedirs(self.article_output_dir, exist_ok=True)
        logger.debug(f"Article output directory: {self.article_output_dir}")

        information_table: StormInformationTable = None
        outline: StormArticle = None
        draft_article: StormArticle = None

        if do_research:
            logger.info("Starten research module")
            information_table = self.run_knowledge_curation_module(
                ground_truth_url=ground_truth_url,
                callback_handler=callback_handler
            )
            logger.info("Research module voltooid")
        
        if do_generate_outline:
            logger.info("Starten outline generation module")
            if information_table is None:
                logger.debug("Laden information table van lokaal bestandssysteem")
                information_table = self._load_information_table_from_local_fs(
                    os.path.join(self.article_output_dir, 'conversation_log.json')
                )
            outline = self.run_outline_generation_module(
                information_table=information_table,
                callback_handler=callback_handler
            )
            logger.info("Outline generation module voltooid")
        
        if do_generate_article:
            logger.info("Starten article generation module")
            if information_table is None:
                logger.debug("Laden information table van lokaal bestandssysteem")
                information_table = self._load_information_table_from_local_fs(
                    os.path.join(self.article_output_dir, 'conversation_log.json')
                )
            if outline is None:
                logger.debug("Laden outline van lokaal bestandssysteem")
                outline = self._load_outline_from_local_fs(
                    topic=topic,
                    outline_local_path=os.path.join(self.article_output_dir, 'storm_gen_outline.txt')
                )
            draft_article = self.run_article_generation_module(
                outline=outline,
                information_table=information_table,
                callback_handler=callback_handler
            )
            logger.info("Article generation module voltooid")
        
        if do_polish_article:
            logger.info("Starten article polishing module")
            if draft_article is None:
                logger.debug("Laden draft article van lokaal bestandssysteem")
                draft_article_path = os.path.join(self.article_output_dir, 'storm_gen_article.txt')
                url_to_info_path = os.path.join(self.article_output_dir, 'url_to_info.json')
                draft_article = self._load_draft_article_from_local_fs(
                    topic=topic,
                    draft_article_path=draft_article_path,
                    url_to_info_path=url_to_info_path
                )
            self.run_article_polishing_module(
                draft_article=draft_article, 
                remove_duplicate=remove_duplicate
            )
            logger.info("Article polishing module voltooid")
        
        self.post_run()
        logger.info(f"STORM pipeline voltooid voor topic: {topic}")