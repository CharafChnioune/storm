from typing import List
from ...interface import Information

# Deze klasse dient als basis voor het beheren van callbacks in de Co-STORM pipeline.
# Het definieert verschillende methoden die worden aangeroepen op specifieke momenten
# tijdens het uitvoeren van de pipeline, waardoor aanpassingen en monitoring mogelijk zijn.
class BaseCallbackHandler:
    """Basis callback handler voor het beheren van callbacks van de Co-STORM pipeline."""

    def on_turn_policy_planning_start(self, **kwargs):
        """Wordt uitgevoerd wanneer de beurtbeleidplanning begint, voordat de richting of het doel voor de volgende gesprekbeurt wordt bepaald."""
        pass

    def on_expert_action_planning_start(self, **kwargs):
        """Wordt uitgevoerd wanneer de planning van expertacties begint, ter voorbereiding op het bepalen van de acties die elke expert moet ondernemen."""
        pass

    def on_expert_action_planning_end(self, **kwargs):
        """Wordt uitgevoerd wanneer de planning van expertacties eindigt, nadat de acties die elke expert moet ondernemen zijn bepaald."""
        pass

    def on_expert_information_collection_start(self, **kwargs):
        """Wordt uitgevoerd wanneer de verzameling van expertinformatie start, begin met het verzamelen van alle nodige gegevens uit geselecteerde bronnen."""
        pass

    def on_expert_information_collection_end(self, info: List[Information], **kwargs):
        """Wordt uitgevoerd wanneer de verzameling van expertinformatie eindigt, nadat alle nodige gegevens uit geselecteerde bronnen zijn verzameld."""
        pass

    def on_expert_utterance_generation_end(self, **kwargs):
        """Wordt uitgevoerd wanneer de generatie van expertuitingen eindigt, voordat reacties of verklaringen van elke expert worden gemaakt."""
        pass

    def on_expert_utterance_polishing_start(self, **kwargs):
        """Wordt uitgevoerd wanneer het polijsten van expertuitingen begint, om de duidelijkheid en samenhang van gegenereerde inhoud te verfijnen en te verbeteren."""
        pass

    def on_mindmap_insert_start(self, **kwargs):
        """Wordt uitgevoerd wanneer het proces van het invoegen van nieuwe informatie in de mindmap start."""
        pass

    def on_mindmap_insert_end(self, **kwargs):
        """Wordt uitgevoerd wanneer het proces van het invoegen van nieuwe informatie in de mindmap eindigt."""
        pass

    def on_mindmap_reorg_start(self, **kwargs):
        """Wordt uitgevoerd wanneer de reorganisatie van de mindmap begint, om de informatiestroom te herstructureren en te optimaliseren."""
        pass

    def on_expert_list_update_start(self, **kwargs):
        """Wordt uitgevoerd wanneer de update van de expertlijst start, om de lijst van actieve experts te wijzigen of te vernieuwen."""
        pass

    def on_article_generation_start(self, **kwargs):
        """Wordt uitgevoerd wanneer het artikelgeneratieproces begint, om de uiteindelijke artikelinhoud samen te stellen en op te maken."""
        pass

    def on_warmstart_update(self, message, **kwargs):
        """Wordt uitgevoerd wanneer het warm start proces een update heeft."""
        pass


# Deze klasse implementeert de BaseCallbackHandler en voegt concrete functionaliteit toe
# door berichten naar de console te printen op verschillende momenten in de pipeline.
# Dit is nuttig voor lokale debugging en monitoring van de Co-STORM pipeline.
class LocalConsolePrintCallBackHandler(BaseCallbackHandler):
    def __init__(self):
        pass

    def on_turn_policy_planning_start(self, **kwargs):
        """Wordt uitgevoerd wanneer de beurtbeleidplanning begint, voordat de richting of het doel voor de volgende gesprekbeurt wordt bepaald."""
        print("Start planning volgende expert; inspecteer mindmap; inspecteer systeemstatus.")

    def on_expert_action_planning_start(self, **kwargs):
        """Wordt uitgevoerd wanneer de planning van expertacties begint, ter voorbereiding op het bepalen van de acties die elke expert moet ondernemen."""
        print("Beoordelen van gespreksgeschiedenis; Beslissen over uiting intentie.")

    def on_expert_information_collection_start(self, **kwargs):
        """Wordt uitgevoerd wanneer de verzameling van expertinformatie start, begin met het zoeken met de zoekmachine en het doorbladeren van verzamelde informatie."""
        print("Start zoeken met de zoekmachine; doorbladeren van verzamelde informatie.")

    def on_expert_information_collection_end(self, info: List[Information], **kwargs):
        """Wordt uitgevoerd wanneer de verzameling van expertinformatie eindigt, nadat alle nodige gegevens uit geselecteerde bronnen zijn verzameld."""
        if info:
            urls = [i.url for i in info]
            information_string = "\n".join([f"Klaar met doorbladeren van {url}" for url in urls])
            print(information_string)

    def on_expert_utterance_generation_end(self, **kwargs):
        """Wordt uitgevoerd wanneer de generatie van expertuitingen eindigt, nadat reacties of verklaringen van elke expert zijn gemaakt."""
        print("Klaar met genereren van uiting op basis van verzamelde informatie.")

    def on_expert_utterance_polishing_start(self, **kwargs):
        """Wordt uitgevoerd wanneer het polijsten van expertuitingen begint, om de duidelijkheid en samenhang van gegenereerde inhoud te verfijnen en te verbeteren."""
        print("Start polijsten van uiting.")

    def on_mindmap_insert_start(self, **kwargs):
        """Wordt uitgevoerd wanneer het proces van het invoegen van nieuwe informatie in de mindmap start."""
        print("Start invoegen van informatie in mindmap.")

    def on_mindmap_insert_end(self, **kwargs):
        """Wordt uitgevoerd wanneer het proces van het invoegen van nieuwe informatie in de mindmap eindigt."""
        print("Klaar met invoegen van informatie in mindmap.")

    def on_mindmap_reorg_start(self, **kwargs):
        """Wordt uitgevoerd wanneer de reorganisatie van de mindmap begint, om de informatiestroom te herstructureren en te optimaliseren."""
        print("Start reorganiseren van mindmap.")

    def on_expert_list_update_start(self, **kwargs):
        """Wordt uitgevoerd wanneer de update van de expertlijst start, om de lijst van actieve experts te wijzigen of te vernieuwen."""
        print("Start updaten van expertenkandidaten.")

    def on_warmstart_update(self, message, **kwargs):
        """Wordt uitgevoerd wanneer het warm start proces een update heeft."""
        print(f"Warm start update: {message}")