# Importeer dspy voor het definiëren van AI-gestuurde modules en signatures
import dspy
import re
from typing import Union


class GenerateExpertGeneral(dspy.Signature):
    """
    Definieert de structuur voor het genereren van een diverse groep experts voor een rondetafelgesprek.
    
    Deze klasse gebruikt dspy.Signature om een gestructureerde input-output relatie te definiëren
    voor het selecteren van experts. Het doel is om een gevarieerde groep sprekers te creëren
    die verschillende perspectieven kunnen bieden op het gegeven onderwerp.

    Aannames:
    - De achtergrondinfo bevat voldoende context om relevante experts te identificeren.
    - Elke expert vertegenwoordigt een uniek perspectief of rol.

    Beperkingen:
    - De output moet strikt het gespecificeerde formaat volgen zonder sprekernamen.
    """

    topic = dspy.InputField(prefix="Onderwerp van interesse:", format=str)
    background_info = dspy.InputField(
        prefix="Achtergrondinformatie over het onderwerp:\n", format=str
    )
    topN = dspy.InputField(prefix="Aantal benodigde sprekers: ", format=str)
    experts = dspy.OutputField(format=str)


class GenerateExpertWithFocus(dspy.Signature):
    """
    Definieert de structuur voor het genereren van een gerichte groep experts voor een rondetafelgesprek.
    
    Deze klasse is vergelijkbaar met GenerateExpertGeneral, maar legt de nadruk op het selecteren
    van experts die direct relevant zijn voor een specifieke focus binnen het hoofdonderwerp.

    Aannames:
    - De gegeven focus is nauw gerelateerd aan het hoofdonderwerp.
    - Experts kunnen tegengestelde standpunten of verschillende belanghebbenden vertegenwoordigen.

    Beperkingen:
    - Geselecteerde sprekers moeten direct verband houden met de gegeven context en scenario.
    - De output moet strikt het gespecificeerde formaat volgen zonder sprekernamen.
    """

    topic = dspy.InputField(prefix="Onderwerp van interesse:", format=str)
    background_info = dspy.InputField(prefix="Achtergrondinformatie:\n", format=str)
    focus = dspy.InputField(prefix="Gespreksfocus: ", format=str)
    topN = dspy.InputField(prefix="Aantal benodigde sprekers: ", format=str)
    experts = dspy.OutputField(format=str)


class GenerateExpertModule(dspy.Module):
    def __init__(self, engine: Union[dspy.dsp.LM, dspy.dsp.HFModel]):
        """
        Initialiseert de module voor het genereren van experts.

        :param engine: Het taalmodel dat gebruikt wordt voor het genereren van experts
        """
        self.engine = engine
        self.generate_expert_general = dspy.Predict(GenerateExpertGeneral)
        self.generate_expert_w_focus = dspy.ChainOfThought(GenerateExpertWithFocus)

    def trim_background(self, background: str, max_words: int = 100):
        """
        Trimt de achtergrondinformatie tot een maximaal aantal woorden.

        Deze methode wordt gebruikt om de input voor het taalmodel te beperken,
        wat helpt bij het focussen op de meest relevante informatie.

        :param background: De originele achtergrondinformatie
        :param max_words: Het maximale aantal woorden (standaard 100)
        :return: De getrimde achtergrondinformatie
        """
        words = background.split()
        cur_len = len(words)
        if cur_len <= max_words:
            return background
        trimmed_words = words[: min(cur_len, max_words)]
        trimmed_background = " ".join(trimmed_words)
        return f"{trimmed_background} [rest van de inhoud weggelaten]."

    def forward(
        self, topic: str, num_experts: int, background_info: str = "", focus: str = ""
    ):
        """
        Genereert een lijst van experts op basis van het gegeven onderwerp en context.

        Deze methode kiest tussen het genereren van algemene experts of experts met een specifieke focus,
        afhankelijk van of er een focus is opgegeven.

        :param topic: Het hoofdonderwerp van het gesprek
        :param num_experts: Het gewenste aantal experts
        :param background_info: Achtergrondinformatie over het onderwerp (optioneel)
        :param focus: Specifieke focus binnen het onderwerp (optioneel)
        :return: Een dspy.Prediction object met de gegenereerde lijst van experts
        """
        with dspy.settings.context(lm=self.engine, show_guidelines=False):
            if not focus:
                # Genereer algemene experts als er geen specifieke focus is
                output = self.generate_expert_general(
                    topic=topic, background_info=background_info, topN=num_experts
                ).experts
            else:
                # Trim de achtergrondinformatie en genereer experts met focus
                background_info = self.trim_background(
                    background=background_info, max_words=100
                )
                output = self.generate_expert_w_focus(
                    topic=topic,
                    background_info=background_info,
                    focus=focus,
                    topN=num_experts,
                ).experts
        
        # Verwerk de output om een lijst van experts te creëren
        output = output.replace("*", "").replace("[", "").replace("]", "")
        expert_list = []
        for s in output.split("\n"):
            match = re.search(r"\d+\.\s*(.*)", s)
            if match:
                expert_list.append(match.group(1))
        expert_list = [expert.strip() for expert in expert_list if expert.strip()]
        
        return dspy.Prediction(experts=expert_list, raw_output=output)