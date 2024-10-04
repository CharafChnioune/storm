import dspy
from typing import Union
from ...dataclass import KnowledgeBase

# Deze module is verantwoordelijk voor het genereren van samenvattingen van de kennisbank
# binnen het Co-STORM framework. Het biedt een overzicht van de besproken onderwerpen
# in een rondetafelgesprek, georganiseerd in hiërarchische secties.

class KnowledgeBaseSummmary(dspy.Signature):
    """Je taak is om een beknopte samenvatting te geven van wat er besproken is in een rondetafelgesprek.
    De inhoud is thematisch georganiseerd in hiërarchische secties.
    Je krijgt deze secties te zien waarbij "#" het niveau van de sectie aangeeft.
    """

    topic = dspy.InputField(prefix="onderwerp: ", format=str)
    structure = dspy.InputField(prefix="Boomstructuur: \n", format=str)
    output = dspy.OutputField(prefix="Geef nu een beknopte samenvatting:\n", format=str)


class KnowledgeBaseSummaryModule(dspy.Module):
    def __init__(self, engine: Union[dspy.dsp.LM, dspy.dsp.HFModel]):
        # Initialiseert de module met een taalmodel engine
        self.engine = engine
        self.gen_summary = dspy.Predict(KnowledgeBaseSummmary)

    def forward(self, knowledge_base: KnowledgeBase):
        # Genereert een samenvatting van de kennisbank
        
        # Haal de hiërarchische structuur van de kennisbank op
        structure = knowledge_base.get_node_hierarchy_string(
            include_indent=False,
            include_full_path=False,
            include_hash_tag=True,
            include_node_content_count=False,
        )
        
        # Gebruik het taalmodel om de samenvatting te genereren
        with dspy.settings.context(lm=self.engine, show_guidelines=False):
            summary = self.gen_summary(
                topic=knowledge_base.topic, structure=structure
            ).output
        
        return summary

# Opmerking: Deze module maakt gebruik van de dspy bibliotheek voor het definiëren van
# de structuur (Signature) en het voorspellen van de samenvatting. Het gebruikt ook een
# aangepaste KnowledgeBase klasse die waarschijnlijk elders in het project is gedefinieerd.
