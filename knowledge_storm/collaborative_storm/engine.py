import dspy
import os
from dataclasses import dataclass, field, asdict
from typing import List, Union, Literal, Optional, Dict

from .modules import collaborative_storm_utils as collaborative_storm_utils
from .modules.callback import BaseCallbackHandler
from .modules.co_storm_agents import (
    SimulatedUser,
    PureRAGAgent,
    Moderator,
    CoStormExpert,
)
from .modules.expert_generation import GenerateExpertModule
from .modules.warmstart_hierarchical_chat import WarmStartModule
from ..dataclass import ConversationTurn, KnowledgeBase
from ..interface import LMConfigs, Agent
from ..logging_wrapper import LoggingWrapper
from ..lm import OpenAIModel, AzureOpenAIModel, TogetherClient, OllamaClient
from ..rm import BingSearch


class CollaborativeStormLMConfigs(LMConfigs):
    """Configuraties voor LLM gebruikt in verschillende delen van Co-STORM.

    Aangezien verschillende onderdelen in het Co-STORM framework verschillende complexiteit hebben,
    gebruiken we verschillende LLM-configuraties om een balans te bereiken tussen kwaliteit en efficiëntie.
    Als er geen specifieke configuratie wordt opgegeven, gebruiken we de standaardinstelling uit het paper.
    """

    def __init__(self):
        self.question_answering_lm = None
        self.discourse_manage_lm = None
        self.utterance_polishing_lm = None
        self.warmstart_outline_gen_lm = None
        self.question_asking_lm = None
        self.knowledge_base_lm = None

    def init(
        self,
        lm_type: Literal["openai", "azure", "together", "ollama"],
        temperature: Optional[float] = 1.0,
        top_p: Optional[float] = 0.9,
    ):
        """
        Initialiseert de LLM-configuraties op basis van het opgegeven type.
        
        Args:
            lm_type: Het type LLM-provider ("openai", "azure", "together" of "ollama")
            temperature: De temperatuur voor sampling (standaard 1.0)
            top_p: De top-p waarde voor sampling (standaard 0.9)
        """
        if lm_type and lm_type == "openai":
            # Configuratie voor OpenAI modellen
            openai_kwargs = {
                "api_key": os.getenv("OPENAI_API_KEY"),
                "api_provider": "openai",
                "temperature": temperature,
                "top_p": top_p,
                "api_base": None,
            }
            self.question_answering_lm = OpenAIModel(
                model="gpt-4o-2024-05-13", max_tokens=1000, **openai_kwargs
            )
            self.discourse_manage_lm = OpenAIModel(
                model="gpt-4o-2024-05-13", max_tokens=500, **openai_kwargs
            )
            self.utterance_polishing_lm = OpenAIModel(
                model="gpt-4o-2024-05-13", max_tokens=2000, **openai_kwargs
            )
            self.warmstart_outline_gen_lm = OpenAIModel(
                model="gpt-4-1106-preview", max_tokens=500, **openai_kwargs
            )
            self.question_asking_lm = OpenAIModel(
                model="gpt-4o-2024-05-13", max_tokens=300, **openai_kwargs
            )
            self.knowledge_base_lm = OpenAIModel(
                model="gpt-4o-2024-05-13", max_tokens=1000, **openai_kwargs
            )
        elif lm_type and lm_type == "azure":
            # Configuratie voor Azure OpenAI modellen
            azure_kwargs = {
                "api_key": os.getenv("AZURE_API_KEY"),
                "temperature": temperature,
                "top_p": top_p,
                "api_base": os.getenv("AZURE_API_BASE"),
                "api_version": os.getenv("AZURE_API_VERSION"),
            }
            self.question_answering_lm = AzureOpenAIModel(
                model="gpt-4o", max_tokens=1000, **azure_kwargs, model_type="chat"
            )
            self.discourse_manage_lm = AzureOpenAIModel(
                model="gpt-4o", max_tokens=500, **azure_kwargs, model_type="chat"
            )
            self.utterance_polishing_lm = AzureOpenAIModel(
                model="gpt-4o", max_tokens=2000, **azure_kwargs, model_type="chat"
            )
            self.warmstart_outline_gen_lm = AzureOpenAIModel(
                model="gpt-4o", max_tokens=300, **azure_kwargs, model_type="chat"
            )
            self.question_asking_lm = AzureOpenAIModel(
                model="gpt-4o", max_tokens=300, **azure_kwargs, model_type="chat"
            )
            self.knowledge_base_lm = AzureOpenAIModel(
                model="gpt-4o", max_tokens=1000, **azure_kwargs, model_type="chat"
            )
        elif lm_type and lm_type == "together":
            # Configuratie voor Together.ai modellen
            together_kwargs = {
                "api_key": os.getenv("TOGETHER_API_KEY"),
                "temperature": temperature,
                "top_p": top_p,
            }
            self.question_answering_lm = TogetherClient(
                model="meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo",
                max_tokens=1000,
                model_type="chat",
                **together_kwargs,
            )
            self.discourse_manage_lm = TogetherClient(
                model="meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo",
                max_tokens=500,
                model_type="chat",
                **together_kwargs,
            )
            self.utterance_polishing_lm = TogetherClient(
                model="meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo",
                max_tokens=2000,
                model_type="chat",
                **together_kwargs,
            )
            self.warmstart_outline_gen_lm = TogetherClient(
                model="meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo",
                max_tokens=500,
                model_type="chat",
                **together_kwargs,
            )
            self.question_asking_lm = TogetherClient(
                model="meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo",
                max_tokens=300,
                model_type="chat",
                **together_kwargs,
            )
            self.knowledge_base_lm = TogetherClient(
                model="meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo",
                max_tokens=1000,
                model_type="chat",
                **together_kwargs,
            )
        elif lm_type and lm_type == "ollama":
            # Configuratie voor Ollama modellen
            ollama_kwargs = {
                "model": "hermes3:8b-llama3.1-fp16",
                "port": 11434,
                "url": "http://localhost",
                "temperature": temperature,
                "top_p": top_p,
                "max_tokens": 1000,
            }
            self.question_answering_lm = OllamaClient(**ollama_kwargs)
            self.discourse_manage_lm = OllamaClient(**ollama_kwargs)
            self.utterance_polishing_lm = OllamaClient(**ollama_kwargs)
            self.warmstart_outline_gen_lm = OllamaClient(**ollama_kwargs)
            self.question_asking_lm = OllamaClient(**ollama_kwargs)
            self.knowledge_base_lm = OllamaClient(**ollama_kwargs)
        else:
            raise Exception(
                "Geen geldige LLM-provider opgegeven. Kan standaard LLM-configuraties niet gebruiken."
            )

    def set_question_answering_lm(self, model: Union[dspy.dsp.LM, dspy.dsp.HFModel]):
        """Stelt het model in voor het beantwoorden van vragen."""
        self.question_answering_lm = model

    def set_discourse_manage_lm(self, model: Union[dspy.dsp.LM, dspy.dsp.HFModel]):
        """Stelt het model in voor het beheren van het gesprek."""
        self.discourse_manage_lm = model

    def set_utterance_polishing_lm(self, model: Union[dspy.dsp.LM, dspy.dsp.HFModel]):
        """Stelt het model in voor het polijsten van uitspraken."""
        self.utterance_polishing_lm = model

    def set_warmstart_outline_gen_lm(self, model: Union[dspy.dsp.LM, dspy.dsp.HFModel]):
        """Stelt het model in voor het genereren van warmstart-outlines."""
        self.warmstart_outline_gen_lm = model

    def set_question_asking_lm(self, model: Union[dspy.dsp.LM, dspy.dsp.HFModel]):
        """Stelt het model in voor het stellen van vragen."""
        self.question_asking_lm = model

    def set_knowledge_base_lm(self, model: Union[dspy.dsp.LM, dspy.dsp.HFModel]):
        """Stelt het model in voor de kennisbank."""
        self.knowledge_base_lm = model

    def collect_and_reset_lm_usage(self):
        """
        Verzamelt gebruiksstatistieken van alle LLM's en reset deze.
        
        Returns:
            dict: Een dictionary met gebruiksstatistieken per LLM.
        """
        lm_usage = {}
        for attr_name in self.__dict__:
            if "_lm" in attr_name and hasattr(
                getattr(self, attr_name), "get_usage_and_reset"
            ):
                usage = getattr(self, attr_name).get_usage_and_reset()
                if any(
                    value["prompt_tokens"] != 0 or value["completion_tokens"] != 0
                    for value in usage.values()
                ):
                    lm_usage[attr_name] = usage
        return lm_usage

    def to_dict(self):
        """
        Zet de CollaborativeStormLMConfigs instantie om naar een dictionary representatie.

        Returns:
            dict: De dictionary representatie van de CollaborativeStormLMConfigs.
        """
        config_dict = {}
        for attr_name in self.__dict__:
            config_dict[attr_name] = getattr(self, attr_name).kwargs
        return config_dict


@dataclass
class RunnerArgument:
    """Argumenten voor het controleren van de STORM Wiki pipeline."""

    topic: str = field(
        metadata={"help": "Onderwerp van het gesprek"},
    )
    retrieve_top_k: int = field(
        default=10,
        metadata={"help": "Haal top k resultaten op voor elke query in de retriever"},
    )
    max_search_queries: int = field(
        default=2,
        metadata={
            "help": "Maximaal aantal zoekqueries om te overwegen voor elke vraag."
        },
    )
    total_conv_turn: int = field(
        default=20,
        metadata={"help": "Maximaal aantal beurten in het gesprek."},
    )
    max_search_thread: int = field(
        default=5,
        metadata={"help": "Maximaal aantal parallelle threads voor de retriever"},
    )
    max_search_queries_per_turn: int = field(
        default=3,
        metadata={"help": "Maximaal aantal zoekqueries om te overwegen in elke beurt."},
    )
    warmstart_max_num_experts: int = field(
        default=3,
        metadata={
            "help": "Max aantal experts in perspectief-geleide QA in warm start proces"
        },
    )
    warmstart_max_turn_per_experts: int = field(
        default=2,
        metadata={"help": "Max aantal beurten per perspectief in warm start proces"},
    )
    warmstart_max_thread: int = field(
        default=3,
        metadata={
            "help": "Max aantal threads voor parallelle perspectief-geleide QA in warm start proces"
        },
    )
    max_thread_num: int = field(
        default=10,
        metadata={
            "help": "Maximaal aantal te gebruiken threads. "
            "Overweeg dit te verlagen als je blijft 'Exceed rate limit' fouten krijgt bij het aanroepen van de LM API."
        },
    )
    max_num_round_table_experts: int = field(
        default=2,
        metadata={"help": "Max aantal actieve experts in ronde tafel discussie."},
    )
    moderator_override_N_consecutive_answering_turn: int = field(
        default=3,
        metadata={
            "help": "Aantal opeenvolgende expert antwoordbeurten voordat de moderator het gesprek overneemt"
        },
    )
    node_expansion_trigger_count: int = field(
        default=10,
        metadata={
            "help": "Trigger node-uitbreiding voor nodes die meer dan N snippets bevatten"
        },
    )
    disable_moderator: bool = field(
        default=False,
        metadata={"help": "Indien True, schakel moderator uit."},
    )
    disable_multi_experts: bool = field(
        default=False,
        metadata={"help": "Indien True, schakel meerdere experts uit."},
    )
    rag_only_baseline_mode: bool = field(
        default=False,
        metadata={"help": "Indien True, schakel over naar rag online baseline modus"},
    )

    def to_dict(self):
        """
        Zet de RunnerArgument instantie om naar een dictionary representatie.

        Returns:
            dict: De dictionary representatie van de RunnerArgument.
        """
        return asdict(self)

    @classmethod
    def from_dict(cls, data):
        """
        Construeert een RunnerArgument instantie uit een dictionary representatie.

        Args:
            data (dict): De dictionary representatie van de RunnerArgument.

        Returns:
            RunnerArgument: De geconstrueerde RunnerArgument instantie.
        """
        return cls(**data)


@dataclass
class TurnPolicySpec:
    """
    Representeert de beleidsspecificaties voor het bepalen van het gedrag van een gespreksbeurt.

    Attributen:
        should_reorganize_knowledge_base (bool):
            Een vlag die aangeeft of de kennisbank moet worden gereorganiseerd na de huidige beurt.

        should_update_experts_list (bool):
            Een vlag die aangeeft of de lijst met experts moet worden bijgewerkt op basis van de gesprekscontext.

        should_polish_utterance (bool):
            Een vlag die aangeeft of de gegenereerde uiting moet worden gepolijst (bijv. verfijnd of herformuleerd) voordat deze in het gesprek wordt gebruikt.

        agent (Agent):
            De `Agent` die verantwoordelijk is voor het genereren van uitingen of antwoorden tijdens de gespreksbeurt.
            Deze agent interacteert met de kennisbank en de gespreksgeschiedenis om antwoorden te produceren.
    """

    should_reorganize_knowledge_base: bool = False
    should_update_experts_list: bool = False
    should_polish_utterance: bool = False
    agent: Agent = None


class DiscourseManager:
    def __init__(
        self,
        logging_wrapper: LoggingWrapper,
        lm_config: CollaborativeStormLMConfigs,
        runner_argument: RunnerArgument,
        rm: dspy.Retrieve,
        callback_handler: BaseCallbackHandler,
    ):
        # parameter beheer
        self.lm_config = lm_config
        self.runner_argument = runner_argument
        self.logging_wrapper = logging_wrapper
        self.callback_handler = callback_handler
        self.rm = rm
        # rol beheer
        self.experts: List[CoStormExpert] = []
        self.simulated_user: SimulatedUser = SimulatedUser(
            topic=self.runner_argument.topic,
            role_name="Gast",
            role_description="",
            intent=None,
            lm_config=self.lm_config,
            runner_argument=self.runner_argument,
            logging_wrapper=self.logging_wrapper,
            callback_handler=self.callback_handler,
        )
        self.pure_rag_agent: PureRAGAgent = PureRAGAgent(
            topic=self.runner_argument.topic,
            role_name="PureRAG",
            role_description="",
            lm_config=self.lm_config,
            runner_argument=self.runner_argument,
            logging_wrapper=self.logging_wrapper,
            rm=self.rm,
            callback_handler=self.callback_handler,
        )
        self.moderator: Moderator = Moderator(
            topic=self.runner_argument.topic,
            role_name="Moderator",
            role_description="",
            lm_config=self.lm_config,
            runner_argument=self.runner_argument,
            logging_wrapper=self.logging_wrapper,
            callback_handler=self.callback_handler,
        )
        self.general_knowledge_provider = CoStormExpert(
            topic=self.runner_argument.topic,
            role_name="Algemene Kennisaanbieder",
            role_description="Focus op het breed behandelen van de basisfeiten over de vraag.",
            lm_config=self.lm_config,
            runner_argument=self.runner_argument,
            logging_wrapper=self.logging_wrapper,
            rm=self.rm,
            callback_handler=self.callback_handler,
        )
        self.generate_expert_module = GenerateExpertModule(
            engine=self.lm_config.discourse_manage_lm
        )
        self.next_turn_moderator_override = False

    def serialize_experts(self) -> List[Dict]:
        """Serialiseert de lijst van experts naar een lijst van dictionaries."""
        return [
            {
                "topic": expert.topic,
                "role_name": expert.role_name,
                "role_description": expert.role_description,
            }
            for expert in self.experts
        ]

    def deserialize_experts(self, data: List[Dict]):
        """Deserialiseert een lijst van dictionaries naar CoStormExpert objecten."""
        for expert_data in data:
            self.experts.append(
                CoStormExpert(
                    topic=expert_data["topic"],
                    role_name=expert_data["role_name"],
                    role_description=expert_data["role_description"],
                    lm_config=self.lm_config,
                    runner_argument=self.runner_argument,
                    logging_wrapper=self.logging_wrapper,
                    rm=self.rm,
                    callback_handler=self.callback_handler,
                )
            )

    def _should_generate_question(
        self, conversation_history: List[ConversationTurn]
    ) -> bool:
        """
        Bepaalt of er een nieuwe vraag moet worden gegenereerd op basis van de gespreksgeschiedenis.
        
        Returns:
            bool: True als er een nieuwe vraag moet worden gegenereerd, anders False.
        """
        consecutive_non_questioning_turn = 0
        for conv_turn in reversed(conversation_history):
            if conv_turn.utterance_type not in [
                "Original Question",
                "Information Request",
            ]:
                consecutive_non_questioning_turn += 1
            else:
                break
        return (
            consecutive_non_questioning_turn
            >= self.runner_argument.moderator_override_N_consecutive_answering_turn
        )

    def _parse_expert_names_to_agent(self, expert_descriptions: Union[str, List[str]]):
        """
        Zet expert beschrijvingen om naar CoStormExpert objecten.
        
        Args:
            expert_descriptions: Een string of lijst van strings met expert beschrijvingen.
        
        Returns:
            List[CoStormExpert]: Een lijst van CoStormExpert objecten.
        """
        if type(expert_descriptions) == str:
            expert_descriptions = [expert_descriptions]
        agents: CoStormExpert = []
        for expert_name in expert_descriptions:
            role_name, role_description = expert_name.split(":")
            role_name = role_name.strip()
            role_description = role_description.strip()
            new_costorm_expert = CoStormExpert(
                topic=self.runner_argument.topic,
                role_name=role_name,
                role_description=role_description,
                lm_config=self.lm_config,
                runner_argument=self.runner_argument,
                logging_wrapper=self.logging_wrapper,
                rm=self.rm,
                callback_handler=self.callback_handler,
            )
            agents.append(new_costorm_expert)
        return agents

    def _update_expert_list_from_utterance(self, focus: str, background_info: str):
        """
        Werkt de lijst van experts bij op basis van de focus en achtergrondinformatie.
        
        Args:
            focus: De huidige focus van het gesprek.
            background_info: Achtergrondinformatie voor het genereren van experts.
        """
        expert_names = self.generate_expert_module(
            topic=self.runner_argument.topic,
            background_info=background_info,
            focus=focus,
            num_experts=self.runner_argument.max_num_round_table_experts,
        ).experts
        self.experts = self._parse_expert_names_to_agent(expert_names)

    def _is_last_turn_questioning(self, conversation_history: List[ConversationTurn]):
        """
        Controleert of de laatste beurt in de gespreksgeschiedenis een vraag was.
        
        Returns:
            bool: True als de laatste beurt een vraag was, anders False.
        """
        return conversation_history and conversation_history[-1].utterance_type in [
            "Original Question",
            "Information Request",
        ]

    def get_next_turn_policy(
        self,
        conversation_history: List[ConversationTurn],
        dry_run=False,
        simulate_user=False,
        simulate_user_intent: str = None,
    ) -> TurnPolicySpec:
        """
        Bepaalt het beleid voor de volgende gespreksbeurt.
        
        Args:
            conversation_history: De geschiedenis van het gesprek.
            dry_run: Indien True, voert een droge run uit zonder de interne staat te wijzigen.
            simulate_user: Indien True, simuleert gebruikersgedrag.
            simulate_user_intent: De gesimuleerde intentie van de gebruiker.
        
        Returns:
            TurnPolicySpec: Het beleid voor de volgende gespreksbeurt.
        """
        next_turn_policy = TurnPolicySpec()
        if simulate_user:
            self.simulated_user.intent = simulate_user_intent
            next_turn_policy.agent = self.simulated_user
        elif self.runner_argument.rag_only_baseline_mode:
            assert self.conversation_history[-1].role == "Guest"
            next_turn_policy.agent = self.pure_rag_agent
        elif (
            not self.runner_argument.disable_moderator
            and self._should_generate_question(conversation_history)
        ):
            next_turn_policy.agent = self.moderator
            next_turn_policy.should_reorganize_knowledge_base = True
        elif self.next_turn_moderator_override:
            next_turn_policy.agent = self.moderator
            if not dry_run:
                self.next_turn_moderator_override = False
        # experts RAG gen
        else:
            next_turn_policy.agent = self.general_knowledge_provider
            if (
                not self._is_last_turn_questioning(conversation_history)
                and not self.runner_argument.disable_multi_experts
            ):
                if dry_run:
                    next_turn_policy.agent = self.experts[0]
                else:
                    next_turn_policy.agent = self.experts.pop(0)
                    self.experts.append(next_turn_policy.agent)
            next_turn_policy.should_update_experts_list = (
                self._is_last_turn_questioning(conversation_history)
                and not self.runner_argument.disable_multi_experts
            )
            next_turn_policy.should_polish_utterance = True
        return next_turn_policy


class CoStormRunner:
    def __init__(
        self,
        lm_config: CollaborativeStormLMConfigs,
        runner_argument: RunnerArgument,
        logging_wrapper: LoggingWrapper,
        rm: Optional[dspy.Retrieve] = None,
        callback_handler: BaseCallbackHandler = None,
    ):
        self.runner_argument = runner_argument
        self.lm_config = lm_config
        self.logging_wrapper = logging_wrapper
        self.callback_handler = callback_handler
        if rm is None:
            self.rm = BingSearch(k=runner_argument.retrieve_top_k)
        else:
            self.rm = rm
        self.conversation_history = []
        self.warmstart_conv_archive = []
        self.knowledge_base = KnowledgeBase(
            topic=self.runner_argument.topic,
            knowledge_base_lm=self.lm_config.knowledge_base_lm,
            node_expansion_trigger_count=self.runner_argument.node_expansion_trigger_count,
        )
        self.discourse_manager = DiscourseManager(
            lm_config=self.lm_config,
            runner_argument=self.runner_argument,
            logging_wrapper=self.logging_wrapper,
            rm=self.rm,
            callback_handler=callback_handler,
        )

    def to_dict(self):
        """
        Zet de CoStormRunner instantie om naar een dictionary representatie.
        
        Returns:
            dict: Een dictionary met de huidige staat van de CoStormRunner.
        """
        return {
            "runner_argument": self.runner_argument.to_dict(),
            "lm_config": self.lm_config.to_dict(),
            "conversation_history": [
                turn.to_dict() for turn in self.conversation_history
            ],
            "warmstart_conv_archive": [
                turn.to_dict() for turn in self.warmstart_conv_archive
            ],
            "experts": self.discourse_manager.serialize_experts(),
            "knowledge_base": self.knowledge_base.to_dict(),
        }

    @classmethod
    def from_dict(cls, data):
        """
        Construeert een CoStormRunner instantie uit een dictionary representatie.
        
        Args:
            data: Een dictionary met de staat van een CoStormRunner.
        
        Returns:
            CoStormRunner: Een nieuwe CoStormRunner instantie.
        """
        # FIXME: gebruikt niet de lm_config data maar naïef de standaardinstelling
        lm_config = CollaborativeStormLMConfigs()
        lm_config.init(lm_type=os.getenv("OPENAI_API_TYPE"))
        costorm_runner = cls(
            lm_config=lm_config,
            runner_argument=RunnerArgument.from_dict(data["runner_argument"]),
            logging_wrapper=LoggingWrapper(lm_config),
        )
        costorm_runner.conversation_history = [
            ConversationTurn.from_dict(turn) for turn in data["conversation_history"]
        ]
        costorm_runner.warmstart_conv_archive = [
            ConversationTurn.from_dict(turn)
            for turn in data.get("warmstart_conv_archive", [])
        ]
        costorm_runner.discourse_manager.deserialize_experts(data["experts"])
        costorm_runner.knowledge_base = KnowledgeBase.from_dict(
            data=data["knowledge_base"],
            knowledge_base_lm=costorm_runner.lm_config.knowledge_base_lm,
            node_expansion_trigger_count=costorm_runner.runner_argument.node_expansion_trigger_count,
        )
        return costorm_runner

    def warm_start(self):
        """
        Warm start co-storm systeem om achtergrond informatie zoeken uit te voeren om een gedeelde conceptuele ruimte met de gebruiker op te bouwen.
        Deze fase is een mini-STORM, waarbij meerdere LLM-agenten met verschillende perspectieven worden ingezet en een meerdere-ronden gesprek wordt gevoerd.
        De kennisbank (d.w.z. mind map) wordt geïnitialiseerd met de verzamelde informatie.

        Het genereert ook een eerste conceptrapport en gebruikt dit om een boeiend en beknopt gesprek te produceren dat aan de gebruiker wordt gepresenteerd
        om bij te blijven met de kennis van het systeem over het onderwerp.
        """
        with self.logging_wrapper.log_pipeline_stage(
            pipeline_stage=f"warm start fase"
        ):
            if not self.runner_argument.rag_only_baseline_mode:
                warm_start_module = WarmStartModule(
                    lm_config=self.lm_config,
                    runner_argument=self.runner_argument,
                    logging_wrapper=self.logging_wrapper,
                    rm=self.rm,
                    callback_handler=self.callback_handler,
                )

                warmstart_conv, warmstart_revised_conv, warmstart_experts = (
                    warm_start_module.initiate_warm_start(
                        topic=self.runner_argument.topic,
                        knowledge_base=self.knowledge_base,
                    )
                )
                self.discourse_manager.experts = (
                    self.discourse_manager._parse_expert_names_to_agent(
                        warmstart_experts
                    )
                )
                self.discourse_manager.next_turn_moderator_override = True
                self.conversation_history = (
                    warmstart_revised_conv if warmstart_revised_conv else warmstart_conv
                )
                self.warmstart_conv_archive = warmstart_conv
                self.knowledge_base.reogranize()
            else:
                if self.knowledge_base is None:
                    self.knowledge_base = KnowledgeBase(
                        topic=self.runner_argument.topic
                    )
                if self.conversation_history is None:
                    self.conversation_history = []
                conv_turn = (
                    self.discourse_manager.pure_rag_agent.generate_topic_background()
                )
                self.conversation_history.append(conv_turn)
                self.knowledge_base.update_from_conv_turn(
                    conv_turn=conv_turn,
                    allow_create_new_node=True,
                    insert_under_root=self.runner_argument.rag_only_baseline_mode,
                )

    def generate_report(self) -> str:
        """
        Genereert een rapport op basis van de georganiseerde verzamelde informatie in de kennisbank (d.w.z. mind map).
        De artikelgeneratie volgt het paradigma in het STORM-paper, waarbij het de mind map-nodes beschouwt als sectienamen en het rapport sectie voor sectie genereert.

        Returns:
            str: Een string die het rapport vertegenwoordigt, met "#" "##" die hiërarchische secties aangeven en [1][2] die referenties aangeven.
        """
        with self.logging_wrapper.log_pipeline_stage("rapport generatie fase"):
            with self.logging_wrapper.log_event(
                "rapport generatie fase: genereer rapport"
            ):
                return self.knowledge_base.to_report()

    def dump_logging_and_reset(self):
        return self.logging_wrapper.dump_logging_and_reset()

    def step(
        self,
        user_utterance: str = "",
        simulate_user: bool = False,
        simulate_user_intent: str = "",
    ) -> ConversationTurn:
        """
        Levert een enkele beurt in de gespreksflow op.

        Deze methode neemt een gebruikersinvoer wanneer de gebruiker ervoor kiest om een uiting in te voegen of genereert de volgende systeemuiting op basis van de huidige gespreksgeschiedenis en gedefinieerde gespreksbeleid.
        Het handelt het bijwerken van de gespreksgeschiedenis af, beheert expertlijsten en interacteert met de kennisbank.
        Daarnaast logt het elke fase van het gesprek voor monitoring en debugging doeleinden.

        Args:
            user_utterance (str, optional): De input geleverd door de gebruiker. Indien verstrekt, wordt deze uiting direct toegevoegd aan de gespreksgeschiedenis en keert terug zonder verdere actie.
            simulate_user (bool, optional): Dit is ontworpen voor automatische experimenten met behulp van een LLM-agent om gebruikersacties te simuleren. Vlag die aangeeft of gebruikersgedrag moet worden gesimuleerd. Wanneer ingesteld op `True`, zal het systeem gebruikersintenties genereren op basis van vooraf gedefinieerde simulatielogica. Standaard is `False`.
            simulate_user_intent (str, optional): Dit is ontworpen voor automatische experimenten met behulp van een LLM-agent om gebruikersacties te simuleren. Specificeert de intentie om te simuleren voor de gebruiker. Dit wordt gebruikt wanneer `simulate_user` `True` is om de reacties van de gesimuleerde gebruiker te sturen.

        Returns:
            ConversationTurn: Een object dat de laatste beurt in het gesprek vertegenwoordigt.

        Workflow:
            1. Gebruikersuiting Afhandeling
                - Als `user_utterance` is verstrekt, wordt het toegevoegd aan de `conversation_history`

            2. Systeemuiting Generatie
                - Als er geen `user_utterance` is verstrekt, gaat de methode verder met het genereren van de volgende systeemuiting.
                - Bepaalt het volgende beurtbeleid door de `discourse_manager` te raadplegen met de huidige gespreksgeschiedenis.
                - Genereert een nieuwe uiting met behulp van de agent gedefinieerd in het beurtbeleid, gebruikmakend van de `knowledge_base` en `conversation_history`.
                - Als het beurtbeleid aangeeft dat de expertlijst moet worden bijgewerkt, werkt het de expertlijst bij op basis van de laatste uitingen.

            4. Kennisbank Update
                - Voegt de nieuwe beurt in in de `knowledge_base`, optioneel toestaan van het creëren van nieuwe nodes of invoegen onder de root op basis van de `rag_only_baseline_mode` vlag.
                - Als het beurtbeleid het specificeert, reorganiseert het de `knowledge_base` om optimale structuur en relevantie te behouden.
        """
        last_conv_turn = self.conversation_history[-1]
        cur_turn_name = f"conv turn: {len(self.conversation_history) + 1}"
        with self.logging_wrapper.log_pipeline_stage(
            pipeline_stage=f"{cur_turn_name} stage"
        ):
            conv_turn = None
            if user_utterance:
                self.discourse_manager.next_turn_moderator_override = False
                conv_turn = ConversationTurn(
                    role="Guest",
                    raw_utterance=user_utterance,
                    utterance_type="Original Question",
                )
                self.conversation_history.append(conv_turn)
            else:
                with self.logging_wrapper.log_event(
                    f"{cur_turn_name}: get turn policy"
                ):
                    if self.callback_handler is not None:
                        self.callback_handler.on_turn_policy_planning_start()
                    turn_policy = self.discourse_manager.get_next_turn_policy(
                        conversation_history=self.conversation_history,
                        simulate_user=simulate_user,
                        simulate_user_intent=simulate_user_intent,
                        dry_run=False,
                    )

                with self.logging_wrapper.log_event(
                    f"{cur_turn_name}: generate utterance"
                ):
                    conv_turn = turn_policy.agent.generate_utterance(
                        knowledge_base=self.knowledge_base,
                        conversation_history=self.conversation_history,
                    )

                if turn_policy.should_update_experts_list:
                    with self.logging_wrapper.log_event(
                        f"{cur_turn_name}: update experts list"
                    ):
                        self.discourse_manager._update_expert_list_from_utterance(
                            focus=last_conv_turn.raw_utterance,
                            background_info=conv_turn.raw_utterance,
                        )

                if conv_turn is not None:
                    self.conversation_history.append(conv_turn)
                    with self.logging_wrapper.log_event(
                        f"{cur_turn_name}: insert into knowledge base"
                    ):
                        if self.callback_handler is not None:
                            self.callback_handler.on_mindmap_insert_start()
                        self.knowledge_base.update_from_conv_turn(
                            conv_turn=conv_turn,
                            allow_create_new_node=True,
                            insert_under_root=self.runner_argument.rag_only_baseline_mode,
                        )
                        if self.callback_handler is not None:
                            self.callback_handler.on_mindmap_insert_end()
                if turn_policy.should_reorganize_knowledge_base:
                    with self.logging_wrapper.log_event(
                        f"{cur_turn_name}: reorganize knowledge base"
                    ):
                        if self.callback_handler is not None:
                            self.callback_handler.on_mindmap_reorg_start()
                        self.knowledge_base.reogranize()
        return conv_turn
