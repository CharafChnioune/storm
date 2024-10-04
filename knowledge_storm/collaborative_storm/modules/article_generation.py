import dspy
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Set, Union

from .collaborative_storm_utils import clean_up_section
from ...dataclass import KnowledgeBase, KnowledgeNode

# Deze klasse is verantwoordelijk voor het genereren van artikelsecties op basis van verzamelde informatie
class ArticleGenerationModule(dspy.Module):
    """Gebruik de informatie verzameld uit het informatiezoekende gesprek om een sectie te schrijven."""

    def __init__(
        self,
        engine: Union[dspy.dsp.LM, dspy.dsp.HFModel],
    ):
        super().__init__()
        # Initialiseer de write_section functie met de WriteSection klasse
        self.write_section = dspy.Predict(WriteSection)
        self.engine = engine

    # Deze methode verzamelt en formatteert de geciteerde informatie
    def _get_cited_information_string(
        self,
        all_citation_index: Set[int],
        knowledge_base: KnowledgeBase,
        max_words: int = 1500,
    ):
        information = []
        cur_word_count = 0
        for index in sorted(list(all_citation_index)):
            info = knowledge_base.info_uuid_to_info_dict[index]
            snippet = info.snippets[0]
            # Formatteer de informatie met index, snippet, vraag en zoekopdracht
            info_text = f"[{index}]: {snippet} (Vraag: {info.meta['question']}. Zoekopdracht: {info.meta['query']})"
            cur_snippet_length = len(info_text.split())
            if cur_snippet_length + cur_word_count > max_words:
                break
            cur_word_count += cur_snippet_length
            information.append(info_text)
        return "\n".join(information)

    # Deze methode genereert een sectie voor een gegeven onderwerp en kennisnode
    def gen_section(
        self, topic: str, node: KnowledgeNode, knowledge_base: KnowledgeBase
    ):
        if node is None or len(node.content) == 0:
            return ""
        # Controleer of de sectie al is gegenereerd en niet opnieuw hoeft te worden gegenereerd
        if (
            node.synthesize_output is not None
            and node.synthesize_output
            and not node.need_regenerate_synthesize_output
        ):
            return node.synthesize_output
        all_citation_index = node.collect_all_content()
        information = self._get_cited_information_string(
            all_citation_index=all_citation_index, knowledge_base=knowledge_base
        )
        # Gebruik de engine om de sectie te genereren
        with dspy.settings.context(lm=self.engine):
            synthesize_output = clean_up_section(
                self.write_section(
                    topic=topic, info=information, section=node.name
                ).output
            )
        node.synthesize_output = synthesize_output
        node.need_regenerate_synthesize_output = False
        return node.synthesize_output

    # Deze methode genereert het volledige artikel door alle secties te combineren
    def forward(self, knowledge_base: KnowledgeBase):
        all_nodes = knowledge_base.collect_all_nodes()
        node_to_paragraph = {}

        # Definieer een functie om paragrafen voor nodes te genereren
        def _node_generate_paragraph(node):
            node_gen_paragraph = self.gen_section(
                topic=knowledge_base.topic, node=node, knowledge_base=knowledge_base
            )
            lines = node_gen_paragraph.split("\n")
            # Verwijder de eerste regel als deze gelijk is aan de naam van de node
            if lines[0].strip().replace("*", "").replace("#", "") == node.name:
                lines = lines[1:]
            node_gen_paragraph = "\n".join(lines)
            path = " -> ".join(node.get_path_from_root())
            return path, node_gen_paragraph

        # Gebruik ThreadPoolExecutor voor parallelle verwerking
        with ThreadPoolExecutor(max_workers=5) as executor:
            # Dien alle taken in
            future_to_node = {
                executor.submit(_node_generate_paragraph, node): node
                for node in all_nodes
            }

            # Verzamel de resultaten zodra ze voltooid zijn
            for future in as_completed(future_to_node):
                path, node_gen_paragraph = future.result()
                node_to_paragraph[path] = node_gen_paragraph

        # Recursieve hulpfunctie om de structuur van het artikel op te bouwen
        def helper(cur_root, level):
            to_return = []
            if cur_root is not None:
                hash_tag = "#" * level + " "
                cur_path = " -> ".join(cur_root.get_path_from_root())
                node_gen_paragraph = node_to_paragraph[cur_path]
                to_return.append(f"{hash_tag}{cur_root.name}\n{node_gen_paragraph}")
                for child in cur_root.children:
                    to_return.extend(helper(child, level + 1))
            return to_return

        to_return = []
        for child in knowledge_base.root.children:
            to_return.extend(helper(child, level=1))

        return "\n".join(to_return)

# Deze klasse definieert de structuur voor het schrijven van een Wikipedia-sectie
class WriteSection(dspy.Signature):
    """Schrijf een Wikipedia-sectie op basis van de verzamelde informatie. Je krijgt het onderwerp, de sectie die je schrijft en relevante informatie.
    Elke informatie wordt geleverd met de ruwe inhoud samen met de vraag en zoekopdracht die tot die informatie hebben geleid.
    Hier is het formaat van je schrijven:
    Gebruik [1], [2], ..., [n] in de regel (bijvoorbeeld, "De hoofdstad van de Verenigde Staten is Washington, D.C.[1][3]."). Je hoeft GEEN Referenties of Bronnen sectie op te nemen om de bronnen aan het einde te vermelden.
    """

    info = dspy.InputField(prefix="De verzamelde informatie:\n", format=str)
    topic = dspy.InputField(prefix="Het onderwerp van de pagina: ", format=str)
    section = dspy.InputField(prefix="De sectie die je moet schrijven: ", format=str)
    output = dspy.OutputField(
        prefix="Schrijf de sectie met juiste inline citaties (Begin met schrijven. Neem de paginatitel, sectienaam niet op en probeer geen andere secties te schrijven. Begin de sectie niet met de onderwerpnaam.):\n",
        format=str,
    )