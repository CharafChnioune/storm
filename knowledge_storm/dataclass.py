import dspy
import numpy as np
import re
import threading
from typing import Set, Dict, List, Optional, Union, Tuple

from .encoder import get_text_embeddings
from .interface import Information


class ConversationTurn:
    """
    Een klasse om een beurt in een gesprek te representeren.

    Attributen:
        role (str): Een korte omschrijving van de rol van de spreker voor de huidige gespreksbeurt.
        raw_utterance (str): De onbewerkte respons gegenereerd door het LM-model zonder verfijnde stijl en toon.
        utterance_type (str): Het type uiting (bijv. verklaring, vraag).
        claim_to_make (Optional[str]): Het punt dat deze uiting probeert te maken. Moet leeg zijn als het uitingstype een vraag is.
        utterance (Optional[str]): De respons gegenereerd door het model met verfijnde stijl en toon. Standaard raw_utterance als niet opgegeven.
        queries (List[str]): De zoekopdrachten gebruikt om informatie te verzamelen voor een onderbouwd antwoord.
        raw_retrieved_info (List['Information']): Een lijst van Information objecten die zijn opgehaald.
        cited_info (Dict[int, 'Information']): Een woordenboek waar de sleutel het citatienummer is en de waarde een Information object.
        role_description (Optional[str]): Een beschrijving van enkele zinnen van de rol. Standaard een lege string als niet opgegeven.
    """

    def __init__(
        self,
        role: str,
        raw_utterance: str,
        utterance_type: str,
        claim_to_make: Optional[str] = None,
        utterance: Optional[str] = None,
        queries: Optional[List[str]] = None,
        raw_retrieved_info: Optional[List[Information]] = None,
        cited_info: Optional[List[Information]] = None,
    ):
        # Initialiseer de attributen van de ConversationTurn
        self.utterance = utterance if utterance is not None else raw_utterance
        self.raw_utterance = raw_utterance
        self.role = role if ":" not in role else role.split(":")[0]
        self.role_description = "" if ":" not in role else role.split(":")[1]
        self.queries = queries if queries is not None else []
        self.raw_retrieved_info = (
            raw_retrieved_info if raw_retrieved_info is not None else []
        )
        self.cited_info = cited_info if cited_info is not None else {}
        self.utterance_type = utterance_type
        self.claim_to_make = claim_to_make if claim_to_make is not None else ""

    def get_all_citation_index(self):
        """
        Haalt alle citatienummers op uit de uiting.
        
        Returns:
            List[int]: Een lijst met alle citatienummers gevonden in de uiting.
        """
        citation_pattern = re.compile(r"\[(\d+)\]")
        return list(map(int, citation_pattern.findall(self.utterance)))

    def to_dict(self):
        """
        Zet het ConversationTurn object om naar een dictionary.
        
        Returns:
            Dict: Een dictionary representatie van het ConversationTurn object.
        """
        raw_retrieved_info = [info.to_dict() for info in self.raw_retrieved_info]
        return {
            "utterance": self.utterance,
            "raw_utterance": self.raw_utterance,
            "role": self.role,
            "role_description": self.role_description,
            "queries": self.queries,
            "utterance_type": self.utterance_type,
            "claim_to_make": self.claim_to_make,
            "raw_retrieved_info": raw_retrieved_info,
            "cited_info": None,
        }

    @classmethod
    def from_dict(cls, conv_turn_dict: Dict):
        """
        Creëert een ConversationTurn object vanuit een dictionary.
        
        Args:
            conv_turn_dict (Dict): Een dictionary met de gegevens voor het ConversationTurn object.
        
        Returns:
            ConversationTurn: Een nieuw ConversationTurn object.
        """
        raw_retrieved_info = [
            Information.from_dict(info) for info in conv_turn_dict["raw_retrieved_info"]
        ]

        return cls(
            utterance=conv_turn_dict["utterance"],
            raw_utterance=conv_turn_dict["raw_utterance"],
            role=f"{conv_turn_dict['role']}: {conv_turn_dict['role_description']}",
            queries=conv_turn_dict["queries"],
            raw_retrieved_info=raw_retrieved_info,
            cited_info=None,
            utterance_type=conv_turn_dict["utterance_type"],
            claim_to_make=conv_turn_dict["claim_to_make"],
        )


class KnowledgeNode:
    """
    Klasse die een knoop in de kennisbasis representeert.

    Attributen:
        name (str): De naam van de knoop.
        content (set): Een set van Information instanties.
        children (list): Een lijst van kind KnowledgeNode instanties.
        parent (KnowledgeNode): De ouderknoop van de huidige knoop.
    """

    def __init__(
        self,
        name: str,
        content: Optional[str] = None,
        parent: Optional["KnowledgeNode"] = None,
        children: Optional[List["KnowledgeNode"]] = None,
        synthesize_output: Optional[str] = None,
        need_regenerate_synthesize_output: bool = True,
    ):
        """
        Initialiseert een KnowledgeNode instantie.

        Args:
            name (str): De naam van de knoop.
            content (list, optional): Een lijst van informatie-uuids. Standaard None.
            parent (KnowledgeNode, optional): De ouderknoop van de huidige knoop. Standaard None.
        """
        self.name = name
        self.content: Set[int] = set(content) if content is not None else set()
        self.children = [] if children is None else children
        self.parent = parent
        self.synthesize_output = synthesize_output
        self.need_regenerate_synthesize_output = need_regenerate_synthesize_output

    def collect_all_content(self):
        """
        Verzamelt alle inhoud van de huidige knoop en zijn nakomelingen.

        Returns:
            Set[int]: Een set met alle inhoud van de huidige knoop en zijn nakomelingen.
        """
        all_content = set(self.content)
        for child in self.children:
            all_content.update(child.collect_all_content())
        return all_content

    def has_child(self, child_node_name: str):
        """
        Controleert of de knoop een kind heeft met de gegeven naam.
        """
        return child_node_name in [child.name for child in self.children]

    def add_child(self, child_node_name: str, duplicate_handling: str = "skip"):
        """
        Voegt een kindknoop toe aan de huidige knoop.
        duplicate_handling (str): Hoe om te gaan met dubbele knopen. Opties zijn "skip", "none", en "raise error".
        """
        if self.has_child(child_node_name):
            if duplicate_handling == "skip":
                for child in self.children:
                    if child.name == child_node_name:
                        return child
            elif duplicate_handling == "raise error":
                raise Exception(
                    f"Insert node error. Node {child_node_name} already exists under its parent node {self.name}."
                )
        child_node = KnowledgeNode(name=child_node_name, parent=self)
        self.children.append(child_node)
        return child_node

    def get_parent(self):
        """
        Geeft de ouderknoop van de huidige knoop terug.

        Returns:
            KnowledgeNode: De ouderknoop van de huidige knoop.
        """
        return self.parent

    def get_children(self):
        """
        Geeft de kinderen van de huidige knoop terug.

        Returns:
            list: Een lijst van kind KnowledgeNode instanties.
        """
        return self.children

    def get_children_names(self):
        """
        Geeft een lijst van kindernamen terug.
        """
        return [child.name for child in self.children]

    def __repr__(self):
        """
        Geeft een string representatie van de KnowledgeNode instantie terug.

        Returns:
            str: String representatie van de KnowledgeNode instantie.
        """
        return f"KnowledgeNode(name={self.name}, content={self.content}, children={len(self.children)})"

    def get_path_from_root(self, root: Optional["KnowledgeNode"] = None):
        """
        Krijg een lijst van namen van de wortel naar deze knoop.

        Returns:
            List[str]: Een lijst van knoopnamen van de wortel naar deze knoop.
        """
        path = []
        current_node = self
        while current_node:
            path.append(current_node.name)
            if root is not None and current_node.name == root.name:
                break
            current_node = current_node.parent
        return path[::-1]

    def insert_information(self, information_index: int):
        """
        Voegt informatie toe aan de knoop.
        
        Args:
            information_index (int): De index van de toe te voegen informatie.
        """
        if information_index not in self.content:
            self.need_regenerate_synthesize_output = True
            self.content.add(information_index)

    def get_all_descendents(self) -> List["KnowledgeNode"]:
        """
        Krijg een lijst van alle afstammende knopen.

        Returns:
            List[KnowledgeNode]: Een lijst van alle afstammende knopen.
        """
        descendents = []

        def collect_descendents(node):
            for child in node.children:
                descendents.append(child)
                collect_descendents(child)

        collect_descendents(self)
        return descendents

    def get_all_predecessors(self) -> List["KnowledgeNode"]:
        """
        Krijg een lijst van alle voorgaande knopen (van huidige knoop naar wortel).

        Returns:
            List[KnowledgeNode]: Een lijst van alle voorgaande knopen.
        """
        predecessors = []
        current_node = self.parent
        while current_node is not None:
            predecessors.append(current_node)
            current_node = current_node.parent
        return predecessors

    def to_dict(self):
        """
        Zet de KnowledgeNode instantie om naar een dictionary representatie.

        Returns:
            dict: De dictionary representatie van de KnowledgeNode.
        """
        return {
            "name": self.name,
            "content": list(self.content),
            "children": [child.to_dict() for child in self.children],
            "parent": self.parent.name if self.parent else None,
            "synthesize_output": self.synthesize_output,
            "need_regenerate_synthesize_output": self.need_regenerate_synthesize_output,
        }

    @classmethod
    def from_dict(cls, data):
        """
        Construeert een KnowledgeNode instantie vanuit een dictionary representatie.

        Args:
            data (dict): De dictionary representatie van de KnowledgeNode.

        Returns:
            KnowledgeNode: De geconstrueerde KnowledgeNode instantie.
        """

        def helper(cls, data, parent_node=None):
            if parent_node is not None:
                assert data["parent"] is not None and data["parent"] == parent_node.name
            node = cls(
                name=data["name"],
                content=data["content"],
                parent=parent_node,
                children=None,
                synthesize_output=data.get("synthesize_output", None),
                need_regenerate_synthesize_output=data.get(
                    "need_regenerate_synthesize_output", True
                ),
            )
            for child_data in data["children"]:
                child_node = helper(cls, child_data, parent_node=node)
                node.children.append(child_node)
            return node

        return helper(cls, data)


class KnowledgeBase:
    """
    Representeert de dynamische, hiërarchische mindmap gebruikt in Co-STORM om het discours te volgen en te organiseren.

    De kennisbasis dient als een gedeelde conceptuele ruimte tussen de gebruiker en het systeem, waardoor effectieve samenwerking mogelijk is door de cognitieve belasting van de gebruiker te verminderen en ervoor te zorgen dat het discours gemakkelijk te volgen is.

    De kennisbasis is gestructureerd als een boom (of mindmap) die verzamelde informatie en concepten dynamisch organiseert naarmate het gesprek vordert.

    De mindmap bestaat uit concepten (knopen) en randen die ouder-kind relaties tussen onderwerpen vertegenwoordigen. Elk concept is gekoppeld aan opgehaalde informatie,
    die onder het meest geschikte concept wordt geplaatst op basis van de bijbehorende vraag en semantische gelijkenis.

    Voor meer details, zie Sectie 3.2 van het Co-STORM paper: https://www.arxiv.org/pdf/2408.15232
    Attributen:
        root (KnowledgeNode): De wortelknoop van de hiërarchische kennisbasis, die het topniveau concept vertegenwoordigt.

    """

    def __init__(
        self,
        topic: str,
        knowledge_base_lm: Union[dspy.dsp.LM, dspy.dsp.HFModel],
        node_expansion_trigger_count: int,
    ):
        """
        Initialiseert een KnowledgeBase instantie.

        Args:
            topic (str): Het onderwerp van de kennisbasis
            expand_node_module (dspy.Module): De module die de kennisbasis ter plaatse organiseert.
                De module moet de kennisbasis als parameter accepteren. Bijv. expand_node_module(self)
            article_generation_module (dspy.Module): De module die een rapport genereert uit de kennisbasis.
                De module moet een string teruggeven. Bijv. report = article_generation_module(self)
        """
        from .collaborative_storm.modules.article_generation import (
            ArticleGenerationModule,
        )
        from .collaborative_storm.modules.information_insertion_module import (
            InsertInformationModule,
            ExpandNodeModule,
        )
        from .collaborative_storm.modules.knowledge_base_summary import (
            KnowledgeBaseSummaryModule,
        )

        self.topic: str = topic

        self.information_insert_module = InsertInformationModule(
            engine=knowledge_base_lm
        )
        self.expand_node_module = ExpandNodeModule(
            engine=knowledge_base_lm,
            information_insert_module=self.information_insert_module,
            node_expansion_trigger_count=node_expansion_trigger_count,
        )
        self.article_generation_module = ArticleGenerationModule(
            engine=knowledge_base_lm
        )
        self.gen_summary_module = KnowledgeBaseSummaryModule(engine=knowledge_base_lm)

        self.root: KnowledgeNode = KnowledgeNode(name="root")
        self.kb_embedding = {
            "hash": hash(""),
            "encoded_structure": np.array([[]]),
            "structure_string": "",
        }
        self.embedding_cache: Dict[str, np.ndarray] = {}
        self.info_uuid_to_info_dict: Dict[int, Information] = {}
        self.info_hash_to_uuid_dict: Dict[int, int] = {}
        self._lock = threading.Lock()

    def to_dict(self):
        """
        Zet de KnowledgeBase om naar een dictionary representatie.
        
        Returns:
            Dict: Een dictionary representatie van de KnowledgeBase.
        """
        info_uuid_to_info_dict = {
            key: value.to_dict() for key, value in self.info_uuid_to_info_dict.items()
        }
        return {
            "topic": self.topic,
            "tree": self.root.to_dict(),
            "info_uuid_to_info_dict": info_uuid_to_info_dict,
            "info_hash_to_uuid_dict": self.info_hash_to_uuid_dict,
        }

    @classmethod
    def from_dict(
        cls,
        data: Dict,
        knowledge_base_lm: Union[dspy.dsp.LM, dspy.dsp.HFModel],
        node_expansion_trigger_count: int,
    ):
        """
        Creëert een KnowledgeBase object vanuit een dictionary.
        
        Args:
            data (Dict): Een dictionary met de gegevens voor de KnowledgeBase.
            knowledge_base_lm: Het taalmodel voor de kennisbasis.
            node_expansion_trigger_count (int): Het aantal triggers voor knoopuitbreiding.
        
        Returns:
            KnowledgeBase: Een nieuw KnowledgeBase object.
        """
        knowledge_base = cls(
            topic=data["topic"],
            knowledge_base_lm=knowledge_base_lm,
            node_expansion_trigger_count=node_expansion_trigger_count,
        )
        knowledge_base.root = KnowledgeNode.from_dict(data["tree"])
        knowledge_base.info_hash_to_uuid_dict = {
            int(key): int(value)
            for key, value in data["info_hash_to_uuid_dict"].items()
        }
        info_uuid_to_info_dict = {
            int(key): Information.from_dict(value)
            for key, value in data["info_uuid_to_info_dict"].items()
        }
        knowledge_base.info_uuid_to_info_dict = info_uuid_to_info_dict
        return knowledge_base

    def get_knowledge_base_structure_embedding(
        self, root: Optional[KnowledgeNode] = None
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Genereert een embedding van de kennisbasisstructuur.
        
        Args:
            root (Optional[KnowledgeNode]): De wortelknoop om vanaf te beginnen. Standaard None.
        
        Returns:
            Tuple[np.ndarray, List[str]]: Een tuple met de gecodeerde structuur en de structuurstrings.
        """
        outline_string = self.get_node_hierarchy_string(
            include_indent=False,
            include_full_path=True,
            include_hash_tag=False,
            root=root,
        )
        outline_string_hash = hash(outline_string)
        if outline_string_hash != self.kb_embedding["hash"]:
            outline_strings: List[str] = outline_string.split("\n")
            cleaned_outline_strings = [
                outline.replace(" -> ", ", ") for outline in outline_strings
            ]
            encoded_outline, _ = get_text_embeddings(
                cleaned_outline_strings, embedding_cache=self.embedding_cache
            )
            self.kb_embedding = {
                "hash": outline_string_hash,
                "encoded_structure": encoded_outline,
                "structure_string": outline_strings,
            }
        return (
            self.kb_embedding["encoded_structure"],
            self.kb_embedding["structure_string"],
        )

    def traverse_down(self, node):
        """
        Doorloopt de boom naar beneden vanaf de gegeven knoop.

        Args:
            node (KnowledgeNode): De knoop om de traversering vanaf te starten.

        Returns:
            list: Een lijst van KnowledgeNode instanties in de volgorde waarin ze werden bezocht.
        """
        nodes = []

        def _traverse(current_node):
            nodes.append(current_node)
            for child in current_node.get_children():
                _traverse(child)

        _traverse(node)
        return nodes

    def traverse_up(self, node):
        """
        Doorloopt de boom naar boven vanaf de gegeven knoop.

        Args:
            node (KnowledgeNode): De knoop om de traversering vanaf te starten.

        Returns:
            list: Een lijst van KnowledgeNode instanties in de volgorde waarin ze werden bezocht.
        """
        nodes = []
        while node is not None:
            nodes.append(node)
            node = node.get_parent()
        return nodes

    def collect_all_nodes(self):
        """
        Verzamelt alle knopen in de kennisbasis.
        
        Returns:
            List[KnowledgeNode]: Een lijst van alle knopen in de kennisbasis.
        """
        nodes = []

        def _collect(node):
            nodes.append(node)
            for child in node.children:
                _collect(child)

        _collect(self.root)
        return nodes

    def insert_node(
        self,
        new_node_name,
        parent_node: Optional[KnowledgeNode] = None,
        duplicate_handling="skip",
    ):
        """
        Voegt een nieuwe knoop toe aan de kennisbasis onder de gespecificeerde ouderknoop.

        Args:
            new_node_name (str): De naam van de nieuwe knoop.
            parent_node_name (str): De naam van de ouderknoop. Als None, wordt de nieuwe knoop onder de wortel ingevoegd.
            duplicate_handling (str): Hoe om te gaan met dubbele knopen. Opties zijn "skip", "none", en "raise error".
        """
        if parent_node is None:
            return self.root.add_child(
                new_node_name, duplicate_handling=duplicate_handling
            )
        else:
            return parent_node.add_child(
                new_node_name, duplicate_handling=duplicate_handling
            )

    def find_node(self, current_node, node_name):
        """
        Zoekt een knoop op naam in de kennisbasis.

        Args:
            current_node (KnowledgeNode): De knoop om de zoektocht vanaf te starten.
            node_name (str): De naam van de te zoeken knoop.

        Returns:
            KnowledgeNode: De knoop met de gespecificeerde naam, of None als niet gevonden.
        """
        if current_node.name == node_name:
            return current_node
        for child in current_node.get_children():
            result = self.find_node(child, node_name)
            if result is not None:
                return result
        return None

    def insert_from_outline_string(self, outline_string, duplicate_handling="skip"):
        """
        Maakt en voegt knopen toe aan de kennisbasis vanuit een string outline.

        Args:
            outline_string (str): De outline string waar elke regel begint met '#' om het niveau aan te geven.
            duplicate_handling (str): Hoe om te gaan met dubbele knopen. Opties zijn "skip", "none", en "raise error".
        """
        last_node_at_level = {}
        for line in outline_string.split("\n"):
            level = line.count("#")
            if level > 0:
                title = line.strip("# ").strip()
                if title.lower() in ["overview", "summary", "introduction"]:
                    continue
                parent_node = None if level == 1 else last_node_at_level.get(level - 1)
                new_node = self.insert_node(
                    new_node_name=title,
                    parent_node=parent_node,
                    duplicate_handling=duplicate_handling,
                )
                last_node_at_level[level] = new_node
                for deeper_level in list(last_node_at_level.keys()):
                    if deeper_level > level:
                        del last_node_at_level[deeper_level]

    def get_node_hierarchy_string(
        self,
        include_indent=False,
        include_full_path=False,
        include_hash_tag=True,
        include_node_content_count=False,
        cited_indices: Optional[List[int]] = None,
        root: Optional[KnowledgeNode] = None,
    ) -> str:
        """
        Genereert een string representatie van de knoophiërarchie.
        
        Args:
            include_indent (bool): Of inspringing moet worden opgenomen.
            include_full_path (bool): Of het volledige pad moet worden opgenomen.
            include_hash_tag (bool): Of hashtags moeten worden opgenomen.
            include_node_content_count (bool): Of het aantal inhoudselementen moet worden opgenomen.
            cited_indices (Optional[List[int]]): Indices van geciteerde knopen.
            root (Optional[KnowledgeNode]): De wortelknoop om vanaf te beginnen.
        
        Returns:
            str: Een string representatie van de knoophiërarchie.
        """

        def find_node_contain_index(node, index):
            """
            Doorloopt de boom naar beneden vanaf de gegeven knoop.

            Args:
                node (KnowledgeNode): De knoop om de traversering vanaf te starten.

            Returns:
                list: Een lijst van KnowledgeNode instanties in de volgorde waarin ze werden bezocht.
            """
            nodes = []

            def _traverse(current_node):
                if current_node is not None and index in current_node.content:
                    nodes.append(current_node)
                for child in current_node.get_children():
                    _traverse(child)

            _traverse(node)
            return nodes

        paths_to_highlight = set()
        nodes_to_include = set()
        if cited_indices is not None:
            for index in cited_indices:
                for cur_node in find_node_contain_index(self.root, index):
                    paths_to_highlight.add(" -> ".join(cur_node.get_path_from_root()))
                    nodes_to_include.add(cur_node)
                    nodes_to_include.update(cur_node.get_all_descendents())
                    predecessors = cur_node.get_all_predecessors()
                    for predecessor in predecessors:
                        nodes_to_include.update(predecessor.children)
                    nodes_to_include.update(predecessors)

        def should_include_node(node):
            if cited_indices is None:
                return True
            return node in nodes_to_include

        def should_omit_child_nodes(node):
            if cited_indices is None:
                return False
            for child in node.children:
                if should_include_node(child):
                    return False
            return True

        def helper(cur_root, level):
            to_return = []
            if cur_root is not None:
                should_include_current_node = should_include_node(cur_root)

                indent = "" if not include_indent else "\t" * (level - 1)
                full_path = " -> ".join(cur_root.get_path_from_root(root=root))
                node_info = cur_root.name if not include_full_path else full_path
                hash_tag = "#" * level + " " if include_hash_tag else ""
                content_count = (
                    f" ({len(cur_root.content)})" if include_node_content_count else ""
                )
                special_note = (
                    ""
                    if cited_indices is None or full_path not in paths_to_highlight
                    else " ⭐"
                )

                if should_include_current_node:
                    to_return.append(
                        f"{indent}{hash_tag}{node_info}{content_count}{special_note}"
                    )
                    if should_omit_child_nodes(cur_root):
                        if len(cur_root.children) > 0:
                            child_indent = indent = (
                                "" if not include_indent else "\t" * (level)
                            )
                            to_return.append(f"{child_indent}...")
                    else:
                        for child in cur_root.children:
                            to_return.extend(helper(child, level + 1))
            return to_return

        to_return = []
        if root is None and self.root is not None:
            for child in self.root.children:
                to_return.extend(helper(child, level=1))
        else:
            to_return.extend(helper(root, level=1))

        return "\n".join(to_return)

    def find_node_by_path(
        self,
        path: str,
        missing_node_handling="abort",
        root: Optional[KnowledgeNode] = None,
    ):
        """
        Geeft de doelknoop terug gegeven een padstring.

        Args:
            path (str): Het pad naar de knoop, met knoopnamen verbonden door " -> ".
            missing_node_handling (str): Hoe om te gaan met ontbrekende knopen. Opties zijn "abort", "create", en "raise error".
            root (Optional[KnowledgeNode]): De wortelknoop om vanaf te beginnen.

        Returns:
            KnowledgeNode: De doelknoop.
        """
        node_names = path.split(" -> ")
        current_node = self.root if root is None else root

        for name in node_names[1:]:
            found_node = next(
                (child for child in current_node.children if child.name == name), None
            )
            if found_node is None:
                if missing_node_handling == "abort":
                    return
                elif missing_node_handling == "create":
                    new_node = current_node.add_child(child_node_name=name)
                    current_node = new_node
                elif missing_node_handling == "raise error":
                    structure = self.get_node_hierarchy_string(
                        include_indent=True,
                        include_full_path=False,
                        include_hash_tag=True,
                    )
                    raise Exception(
                        f"Insert information error. Unable to find node {{{name}}} under {{{current_node.name}}}\n{structure}"
                    )
            else:
                current_node = found_node
        return current_node

    def insert_information(
        self,
        path: str,
        information: Information,
        missing_node_handling="abort",
        root: Optional[KnowledgeNode] = None,
    ):
        """
        Voegt informatie toe aan de kennisbasis op het gespecificeerde pad.

        Args:
            path (str): Het plaatsingspad, verbonden door " -> " die de namen van knopen koppelt.
            information (Information): De toe te voegen informatie.
            missing_node_handling (str): Hoe om te gaan met ontbrekende knopen. Opties zijn "abort", "create" en "raise error".
        Return:
            uuid van de ingevoegde informatie
        """
        with self._lock:
            target_node: KnowledgeNode = self.find_node_by_path(
                path=path, missing_node_handling=missing_node_handling, root=root
            )
            information_hash = hash(information)
            if information.citation_uuid == -1:
                info_citation_uuid = self.info_hash_to_uuid_dict.get(
                    information_hash, len(self.info_hash_to_uuid_dict) + 1
                )
                information.citation_uuid = info_citation_uuid
                self.info_hash_to_uuid_dict[information_hash] = info_citation_uuid
                self.info_uuid_to_info_dict[info_citation_uuid] = information
            if target_node is not None:
                self.info_uuid_to_info_dict[information.citation_uuid].meta[
                    "placement"
                ] = " -> ".join(target_node.get_path_from_root())
                target_node.insert_information(information.citation_uuid)

    def trim_empty_leaf_nodes(self):
        """
        Snoeit alle bladknopen die geen inhoud hebben. Doet dit iteratief totdat alle bladknopen ten minste één inhoudselement hebben.
        """

        def trim_node(node):
            if not node.children and not node.content:
                return True
            node.children = [child for child in node.children if not trim_node(child)]
            return not node.children and not node.content

        # Start het snoeiproces vanaf de wortel
        while True:
            before_trim = len(self.get_all_leaf_nodes())
            trim_node(self.root)
            after_trim = len(self.get_all_leaf_nodes())
            if before_trim == after_trim:
                break

    def get_all_leaf_nodes(self):
        """
        Hulpfunctie om alle bladknopen te verkrijgen.

        Returns:
            List[KnowledgeNode]: Een lijst van alle bladknopen in de kennisbasis.
        """
        leaf_nodes = []

        def find_leaf_nodes(node):
            if not node.children:
                leaf_nodes.append(node)
            for child in node.children:
                find_leaf_nodes(child)

        find_leaf_nodes(self.root)
        return leaf_nodes

    def merge_single_child_nodes(self):
        """
        Voegt de inhoud van een knoop samen met zijn enige kind en verwijdert de kindknoop.
        Doet dit iteratief van bladknopen terug naar de wortel.
        """

        def merge_node(node):
            # Voeg eerst recursief kinderen samen
            for child in node.children:
                merge_node(child)

            # Als de knoop precies één kind heeft, voeg de inhoud samen met het kind en verwijder het kind
            if len(node.children) == 1:
                single_child = node.children[0]
                node.content.update(single_child.content)
                node.children = single_child.children
                for grandchild in node.children:
                    grandchild.parent = node

        merge_node(self.root)

    def update_all_info_path(self):
        def _helper(node):
            for citation_idx in node.content:
                self.info_uuid_to_info_dict[citation_idx].meta["placement"] = (
                    " -> ".join(node.get_path_from_root())
                )
            for child in node.children:
                _helper(child)

        _helper(self.root)

    def update_from_conv_turn(
        self,
        conv_turn: ConversationTurn,
        allow_create_new_node: bool = False,
        insert_under_root: bool = False,
    ):
        """
        Werkt de kennisbasis bij met informatie uit een gesprekbeurt.

        Args:
            conv_turn (ConversationTurn): De gesprekbeurt om te verwerken.
            allow_create_new_node (bool): Of het maken van nieuwe knopen is toegestaan.
            insert_under_root (bool): Of de informatie direct onder de wortel moet worden ingevoegd.
        """
        if conv_turn is None:
            return
        info_to_insert = list(conv_turn.cited_info.values())
        if insert_under_root:
            for info in info_to_insert:
                self.insert_information(path=self.root.name, information=info)
        else:
            self.information_insert_module(
                knowledge_base=self,
                information=info_to_insert,
                allow_create_new_node=allow_create_new_node,
            )
        old_to_new_citation_idx_mapping = {
            old_idx: info.citation_uuid
            for old_idx, info in conv_turn.cited_info.items()
        }

        # Werk citatie-indices bij in de uitingen
        for old_idx, new_idx in old_to_new_citation_idx_mapping.items():
            conv_turn.utterance = conv_turn.utterance.replace(
                f"[{old_idx}]", f"[_{new_idx}_]"
            )
            conv_turn.raw_utterance = conv_turn.raw_utterance.replace(
                f"[{old_idx}]", f"[_{new_idx}_]"
            )
        for _, new_idx in old_to_new_citation_idx_mapping.items():
            conv_turn.utterance = conv_turn.utterance.replace(
                f"[_{new_idx}_]", f"[{new_idx}]"
            )
            conv_turn.utterance = conv_turn.utterance.replace("[-1]", "")
            conv_turn.raw_utterance = conv_turn.raw_utterance.replace(
                f"[_{new_idx}_]", f"[{new_idx}]"
            )
            conv_turn.raw_utterance = conv_turn.raw_utterance.replace("[-1]", "")
        conv_turn.cited_info = None

    def get_knowledge_base_summary(self):
        """
        Genereert een samenvatting van de kennisbasis.

        Returns:
            str: Een samenvatting van de kennisbasis.
        """
        return self.gen_summary_module(self)

    def reogranize(self):
        """
        Reorganiseert de kennisbasis door twee hoofdprocessen: top-down uitbreiding en bottom-up opschoning.

        Het reorganisatieproces zorgt ervoor dat de kennisbasis goed gestructureerd en relevant blijft naarmate nieuwe informatie wordt toegevoegd. Het bestaat uit de volgende stappen:
        1. Top-Down Uitbreiding: Breidt knopen uit die aanzienlijke hoeveelheden informatie hebben verzameld door subonderwerpen te creëren,
           waardoor elk concept specifiek en beheersbaar blijft.
        2. Bottom-Up Opschoning: Schoont de kennisbasis op door lege bladknopen (knopen zonder ondersteunende informatie) te verwijderen
           en knopen samen te voegen die slechts één kind hebben, waardoor de structuur wordt vereenvoudigd en de duidelijkheid behouden blijft.
        """
        # Voorbewerking
        self.trim_empty_leaf_nodes()
        self.merge_single_child_nodes()
        # Breid knopen uit
        self.expand_node_module(knowledge_base=self)
        # Opschoning
        self.trim_empty_leaf_nodes()
        self.merge_single_child_nodes()
        self.update_all_info_path()

    def to_report(self):
        """
        Genereert een rapport van de kennisbasis.

        Returns:
            str: Een rapport gegenereerd uit de kennisbasis.
        """
        return self.article_generation_module(knowledge_base=self)