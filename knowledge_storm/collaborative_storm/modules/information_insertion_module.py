import dspy
import numpy as np
import re
import traceback

from concurrent.futures import ThreadPoolExecutor, as_completed
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Union, Dict, Optional

from .collaborative_storm_utils import trim_output_after_hint
from ...dataclass import KnowledgeNode, KnowledgeBase
from ...encoder import get_text_embeddings
from ...interface import Information

# Deze module behandelt het invoegen van informatie in een kennisbank binnen het Co-STORM framework.
# Het gebruikt een boomstructuur om informatie te organiseren op basis van semantische gelijkenis.

class InsertInformation(dspy.Signature):
    """Je taak is om de gegeven informatie in de kennisbank in te voegen. De kennisbank is een boomgebaseerde datastructuur om verzamelde informatie te organiseren. Elke kennisnode bevat informatie afgeleid van semantisch vergelijkbare vragen of intenties.
    Om de beste plaatsing van de informatie te bepalen, zul je laag voor laag door deze boomstructuur worden genavigeerd.
    Je krijgt de vraag en zoekopdracht te zien die tot deze informatie hebben geleid, evenals de boomstructuur.

    De output moet strikt een van de onderstaande opties volgen zonder andere informatie:
    - 'insert': om de informatie onder de huidige node te plaatsen.
    - 'step: [naam van kindnode]': om naar een specifieke kindnode te gaan.
    - 'create: [naam van nieuwe kindnode]': om een nieuwe kindnode te maken en de info eronder in te voegen.

    Voorbeeldoutputs:
    - insert
    - step: node2
    - create: node3
    """

    intent = dspy.InputField(
        prefix="Vraag en zoekopdracht die tot deze info leiden: ", format=str
    )
    structure = dspy.InputField(prefix="Boomstructuur: \n", format=str)
    choice = dspy.OutputField(prefix="Keuze:\n", format=str)


class InsertInformationCandidateChoice(dspy.Signature):
    """Je taak is om de gegeven informatie in de kennisbank in te voegen. De kennisbank is een boomgebaseerde datastructuur om verzamelde informatie te organiseren. Elke kennisnode bevat informatie afgeleid van semantisch vergelijkbare vragen of intenties.
    Je krijgt de vraag en zoekopdracht te zien die tot deze informatie hebben geleid, en kandidaat-keuzes voor plaatsing. In deze keuzes geeft -> de ouder-kindrelatie aan. Merk op dat een redelijke keuze mogelijk niet in deze opties staat.

    Als er een redelijke keuze bestaat, output dan "Beste plaatsing: [keuze-index]"; anders, output "Geen redelijke keuze".
    """

    intent = dspy.InputField(
        prefix="Vraag en zoekopdracht die tot deze info leiden: ", format=str
    )
    choices = dspy.InputField(prefix="Kandidaat-plaatsingen:\n", format=str)
    decision = dspy.OutputField(prefix="Beslissing:\n", format=str)


class InsertInformationModule(dspy.Module):
    def __init__(self, engine: Union[dspy.dsp.LM, dspy.dsp.HFModel]):
        self.engine = engine
        self.insert_info = dspy.ChainOfThought(InsertInformation)
        self.candidate_choosing = dspy.Predict(InsertInformationCandidateChoice)

    def _construct_intent(self, question: str, query: str):
        # Construeert een intentstring op basis van de vraag en zoekopdracht
        intent = ""
        if query == "Not applicable":
            return question
        if question:
            intent += f"Vraag: {question}\n"
        if query:
            intent += f"Zoekopdracht: {query}\n"
        if not intent:
            intent = "Niet beschikbaar."
        return intent

    def _get_navigation_choice(
        self, knowledge_node: KnowledgeNode, question: str, query: str
    ):
        # Bepaalt de navigatiekeuze voor het invoegen van informatie
        intent = self._construct_intent(question, query)
        structure = f"Huidige Node: {knowledge_node.name}\n"
        child_names = ", ".join(knowledge_node.get_children_names())
        if child_names:
            structure += f"Kindnodes: {child_names}"
        navigated_path = " -> ".join(knowledge_node.get_path_from_root())
        structure += f"Pad dat je hebt genavigeerd: {navigated_path}"

        with dspy.settings.context(lm=self.engine):
            predicted_action = self.insert_info(
                intent=intent, structure=structure
            ).choice

        # Parseert de voorspelde actie
        cleaned_predicted_action = trim_output_after_hint(
            predicted_action, "Keuze:"
        ).strip()
        cleaned_predicted_action = cleaned_predicted_action.strip("-").strip()
        if cleaned_predicted_action.startswith("insert"):
            return "insert", ""
        elif cleaned_predicted_action.startswith("step:"):
            node_name = trim_output_after_hint(cleaned_predicted_action, "step:")
            return "step", node_name
        elif cleaned_predicted_action.startswith("create:"):
            node_name = trim_output_after_hint(cleaned_predicted_action, "create:")
            return "create", node_name
        raise Exception(
            f"Ongedefinieerde voorspelde actie in kennisnavigatie. {predicted_action}"
        )

    def layer_by_layer_navigation_placement(
        self,
        knowledge_base: KnowledgeBase,
        question: str,
        query: str,
        allow_create_new_node: bool = False,
        root: Optional[KnowledgeNode] = None,
    ):
        # Navigeert laag voor laag door de kennisbank om de juiste plaatsing te vinden
        current_node: KnowledgeNode = knowledge_base.root if root is None else root

        while True:
            action_type, node_name = self._get_navigation_choice(
                knowledge_node=current_node, question=question, query=query
            )
            if action_type == "insert":
                return dspy.Prediction(
                    information_placement=" -> ".join(
                        current_node.get_path_from_root(root)
                    ),
                    note="None",
                )
            elif action_type == "step":
                for child in current_node.children:
                    if child.name == node_name:
                        current_node = child
                        break
                else:
                    raise ValueError(f"Kindnode met naam {node_name} niet gevonden.")
            elif action_type == "create":
                placement_path = current_node.get_path_from_root(root)
                if allow_create_new_node:
                    placement_path.append(node_name)
                    note = f"maak nieuwe node: {{{node_name}}} onder {{{current_node.name}}}"
                else:
                    note = f"poging om nieuwe node te maken: {{{node_name}}} onder {{{current_node.name}}}"
                return dspy.Prediction(
                    information_placement=" -> ".join(placement_path), note=note
                )
            else:
                raise ValueError(f"Onbekend actietype: {action_type}")

    def _get_sorted_embed_sim_section(
        self,
        encoded_outline: np.ndarray,
        outlines: List[str],
        question: str,
        query: str,
    ):
        # Sorteert secties op basis van embedding-gelijkenis
        if encoded_outline is not None and encoded_outline.size > 0:
            encoded_query, token_usage = get_text_embeddings(f"{question}, {query}")
            sim = cosine_similarity([encoded_query], encoded_outline)[0]
            sorted_indices = np.argsort(sim)
            sorted_outlines = np.array(outlines)[sorted_indices[::-1]]
            return sorted_outlines
        else:
            return outlines

    def _parse_selected_index(self, string: str):
        # Parseert de geselecteerde index uit een string
        match = re.search(r"\[(\d+)\]", string)
        if match:
            return int(match.group(1))
        try:
            return int(string.strip())
        except:
            pass
        return None

    def choose_candidate_from_embedding_ranking(
        self,
        question: str,
        query: str,
        encoded_outlines: np.ndarray,
        outlines: List[str],
        top_N_candidates: int = 5,
    ):
        # Kiest een kandidaat op basis van embedding-ranking
        sorted_candidates = self._get_sorted_embed_sim_section(
            encoded_outlines, outlines, question, query
        )
        considered_candidates = sorted_candidates[
            : min(len(sorted_candidates), top_N_candidates)
        ]
        choices_string = "\n".join(
            [
                f"{idx + 1}: {candidate}"
                for idx, candidate in enumerate(considered_candidates)
            ]
        )
        with dspy.settings.context(lm=self.engine, show_guidelines=False):
            decision = self.candidate_choosing(
                intent=self._construct_intent(question=question, query=query),
                choices=choices_string,
            ).decision
            decision = trim_output_after_hint(decision, hint="Beslissing:")
            if "Beste plaatsing:" in decision:
                decision = trim_output_after_hint(decision, hint="Beste plaatsing:")
                selected_index = self._parse_selected_index(decision)
                if selected_index is not None:
                    selected_index = selected_index - 1
                    if selected_index < len(sorted_candidates) and selected_index >= 0:
                        return dspy.Prediction(
                            information_placement=sorted_candidates[selected_index],
                            note=f"Keuze uit:\n{considered_candidates}",
                        )
            return None

    def _info_list_to_intent_mapping(self, information_list: List[Information]):
        # Maakt een mapping van intenties naar plaatsingen
        intent_to_placement_dict = {}
        for info in information_list:
            intent = (info.meta.get("question", ""), info.meta.get("query", ""))
            if intent not in intent_to_placement_dict:
                intent_to_placement_dict[intent] = None
        return intent_to_placement_dict

    def forward(
        self,
        knowledge_base: KnowledgeBase,
        information: Union[Information, List[Information]],
        allow_create_new_node: bool = False,
        max_thread: int = 5,
        insert_root: Optional[KnowledgeNode] = None,
        skip_candidate_from_embedding: bool = False,
    ):
        # Hoofdmethode voor het invoegen van informatie in de kennisbank
        if not isinstance(information, List):
            information = [information]
        intent_to_placement_dict: Dict = self._info_list_to_intent_mapping(
            information_list=information
        )

        def process_intent(question: str, query: str):
            # Verwerkt een enkele intent
            candidate_placement = None
            try:
                if not skip_candidate_from_embedding:
                    candidate_placement = self.choose_candidate_from_embedding_ranking(
                        question=question,
                        query=query,
                        encoded_outlines=encoded_outlines,
                        outlines=outlines,
                        top_N_candidates=8,
                    )
                if candidate_placement is None:
                    candidate_placement = self.layer_by_layer_navigation_placement(
                        knowledge_base=knowledge_base,
                        question=question,
                        query=query,
                        allow_create_new_node=allow_create_new_node,
                        root=insert_root,
                    )
                return (question, query), candidate_placement
            except Exception as e:
                print(traceback.format_exc())
                return (question, query), None

        def insert_info_to_kb(info, placement_prediction):
            # Voegt informatie in de kennisbank in
            if placement_prediction is not None:
                missing_node_handling = (
                    "raise error" if not allow_create_new_node else "create"
                )
                knowledge_base.insert_information(
                    path=placement_prediction.information_placement,
                    information=info,
                    missing_node_handling=missing_node_handling,
                    root=insert_root,
                )

        encoded_outlines, outlines = (
            knowledge_base.get_knowledge_base_structure_embedding(root=insert_root)
        )
        to_return = []
        if not allow_create_new_node:
            # Gebruikt multi-threading omdat de kennisbankstructuur niet verandert
            with ThreadPoolExecutor(max_workers=max_thread) as executor:
                futures = {
                    executor.submit(process_intent, question, query): (question, query)
                    for (question, query) in intent_to_placement_dict
                }

                for future in as_completed(futures):
                    (question, query), candidate_placement = future.result()
                    intent_to_placement_dict[(question, query)] = candidate_placement
            # Koppelt plaatsingen terug aan elke informatie
            for info in information:
                intent = (info.meta.get("question", ""), info.meta.get("query", ""))
                placement_prediction = intent_to_placement_dict.get(intent, None)
                insert_info_to_kb(info, placement_prediction)
                to_return.append((info, placement_prediction))
            return to_return
        else:
            # Gebruikt sequentiële invoeg omdat de kennisbankstructuur kan veranderen
            for question, query in intent_to_placement_dict:
                encoded_outlines, outlines = (
                    knowledge_base.get_knowledge_base_structure_embedding(
                        root=insert_root
                    )
                )
                _, placement_prediction = process_intent(question=question, query=query)
                intent_to_placement_dict[(question, query)] = placement_prediction

            for info in information:
                intent = (info.meta.get("question", ""), info.meta.get("query", ""))
                placement_prediction = intent_to_placement_dict.get(intent, None)
                insert_info_to_kb(info, placement_prediction)
                to_return.append((info, placement_prediction))
            return to_return


class ExpandSection(dspy.Signature):
    """Je taak is om een sectie in de mindmap uit te breiden door nieuwe subsecties onder de gegeven sectie te maken.
    Je krijgt een lijst met vragen en zoekopdrachten die gebruikt zijn om informatie te verzamelen.
    De output moet subsectienamen zijn waarbij elke sectie dient als een coherente en thematische organisatie van informatie en bijbehorende citatienummers. Deze subsectienamen moeten bij voorkeur beknopt en precies zijn.
    De output volgt het onderstaande formaat:
    subsectie 1
    subsectie 2
    subsectie 3
    """

    section = dspy.InputField(prefix="De sectie die je moet uitbreiden: ", format=str)
    info = dspy.InputField(prefix="De verzamelde informatie:\n", format=str)
    output = dspy.OutputField(
        prefix="Geef nu de uitgebreide subsectienamen (Als er geen noodzaak is om de huidige sectie uit te breiden omdat deze al een goede organisatie biedt, output dan None):\n",
        format=str,
    )


class ExpandNodeModule(dspy.Module):
    def __init__(
        self,
        engine: Union[dspy.dsp.LM, dspy.dsp.HFModel],
        information_insert_module: dspy.Module,
        node_expansion_trigger_count: int,
    ):
        self.engine = engine
        self.expand_section = dspy.Predict(ExpandSection)
        self.information_insert_module = information_insert_module
        self.node_expansion_trigger_count = node_expansion_trigger_count

    def _get_cited_info_meta_string(self, node, knowledge_base):
        # Haalt meta-informatie op van geciteerde informatie in een node
        meta_string = set()
        for index in sorted(list(node.content)):
            info = knowledge_base.info_uuid_to_info_dict[index]
            intent = f"Vraag: {info.meta['question']}\nZoekopdracht: {info.meta['query']}"
            meta_string.add(intent)

        return "\n\n".join(meta_string)

    def _get_expand_subnode_names(self, node, knowledge_base):
        # Genereert namen voor nieuwe subnodes
        information = self._get_cited_info_meta_string(node, knowledge_base)
        node_path = node.get_path_from_root()
        with dspy.settings.context(lm=self.engine, show_guidelines=False):
            output = self.expand_section(section=node_path, info=information).output
        subsections = []
        if "\n" in output and output != "None":
            subsections = output.split("\n")
            # Verwijdert eventuele getallen gevolgd door een punt en een spatie, een leidende streep,
            # of een specifieke hint aan het begin van de string
            subsections = [
                re.sub(r"^\d+\.\s|-|" + re.escape(node.name), "", text)
                .replace("*", "")
                .strip()
                for text in subsections
            ]
        return subsections

    def _find_first_node_to_expand(
        self, root: KnowledgeNode, expanded_nodes: List[KnowledgeNode]
    ):
        # Vindt de eerste node die uitgebreid moet worden
        if root is None:
            return None
        if (
            root not in expanded_nodes
            and len(root.content) >= self.node_expansion_trigger_count
        ):
            return root
        for child in root.children:
            to_return = self._find_first_node_to_expand(
                root=child, expanded_nodes=expanded_nodes
            )
            if to_return is not None:
                return to_return
        return None

    def _expand_node(self, node: KnowledgeNode, knowledge_base: KnowledgeBase):
        # Breidt een node uit met nieuwe subnodes
        subsection_names = self._get_expand_subnode_names(node, knowledge_base)
        if len(subsection_names) <= 1:
            return
        # Maakt nieuwe nodes
        for subsection_name in subsection_names:
            # Verwijdert citaathaakjes in de subsectienaam
            subsection_name = re.sub(r"\[.*?\]", "", subsection_name)
            knowledge_base.insert_node(new_node_name=subsection_name, parent_node=node)
        # Reset originele informatie plaatsing
        original_cited_index = node.content
        original_cited_information = [
            knowledge_base.info_uuid_to_info_dict[index]
            for index in original_cited_index
        ]
        node.content = set()
        # Voegt opnieuw in onder uitgebreide sectie
        self.information_insert_module(
            knowledge_base=knowledge_base,
            information=original_cited_information,
            allow_create_new_node=False,
            insert_root=node,
        )

    def forward(self, knowledge_base: KnowledgeBase):
        # Hoofdmethode voor het uitbreiden van nodes in de kennisbank
        expanded_nodes = []
        while True:
            node_to_expand = self._find_first_node_to_expand(
                root=knowledge_base.root, expanded_nodes=expanded_nodes
            )
            if node_to_expand is None:
                break
            self._expand_node(node=node_to_expand, knowledge_base=knowledge_base)
            expanded_nodes.append(node_to_expand)