import logging
import re
from typing import Union, List

import dspy
import requests
from bs4 import BeautifulSoup

# Functie om de titel en inhoudsopgave van een Wikipedia-pagina op te halen
def get_wiki_page_title_and_toc(url):
    """Haal de hoofdtitel en inhoudsopgave op van een Wikipedia-pagina URL."""

    response = requests.get(url)
    soup = BeautifulSoup(response.content, "html.parser")

    # Haal de hoofdtitel op uit de eerste h1-tag
    main_title = soup.find("h1").text.replace("[edit]", "").strip().replace("\xa0", " ")

    toc = ""
    levels = []
    excluded_sections = {
        "Contents",
        "See also",
        "Notes",
        "References",
        "External links",
    }

    # Begin met verwerken vanaf h2 om de hoofdtitel uit de inhoudsopgave te sluiten
    for header in soup.find_all(["h2", "h3", "h4", "h5", "h6"]):
        level = int(header.name[1])  # Haal het numerieke deel uit de header-tag (bijv. '2' van 'h2')
        section_title = header.text.replace("[edit]", "").strip().replace("\xa0", " ")
        if section_title in excluded_sections:
            continue

        # Pas de niveaus aan op basis van de huidige header
        while levels and level <= levels[-1]:
            levels.pop()
        levels.append(level)

        # Voeg de sectietitel toe aan de inhoudsopgave met juiste inspringing
        indentation = "  " * (len(levels) - 1)
        toc += f"{indentation}{section_title}\n"

    return main_title, toc.strip()

# Klasse voor het vinden van gerelateerde onderwerpen
class FindRelatedTopic(dspy.Signature):
    """Ik schrijf een Wikipedia-pagina voor een onderstaand genoemd onderwerp. Identificeer en adviseer enkele Wikipedia-pagina's over nauw verwante onderwerpen. Ik zoek naar voorbeelden die inzicht geven in interessante aspecten die vaak met dit onderwerp worden geassocieerd, of voorbeelden die me helpen de typische inhoud en structuur te begrijpen die in Wikipedia-pagina's voor vergelijkbare onderwerpen worden opgenomen.
    Geef de URL's op aparte regels weer."""

    topic = dspy.InputField(prefix="Onderwerp van interesse:", format=str)
    related_topics = dspy.OutputField(format=str)

# Klasse voor het genereren van persona's
class GenPersona(dspy.Signature):
    """Je moet een groep Wikipedia-redacteuren selecteren die samenwerken om een uitgebreid artikel over het onderwerp te maken. Elk van hen vertegenwoordigt een ander perspectief, rol of affiliatie gerelateerd aan dit onderwerp. Je kunt andere Wikipedia-pagina's van gerelateerde onderwerpen gebruiken als inspiratie. Voeg voor elke redacteur een beschrijving toe van waar ze zich op zullen richten.
    Geef je antwoord in het volgende formaat: 1. korte samenvatting van redacteur 1: beschrijving\n2. korte samenvatting van redacteur 2: beschrijving\n..."""

    topic = dspy.InputField(prefix="Onderwerp van interesse:", format=str)
    examples = dspy.InputField(
        prefix="Wiki-pagina-outlines van gerelateerde onderwerpen ter inspiratie:\n", format=str
    )
    personas = dspy.OutputField(format=str)

# Klasse voor het creëren van een schrijver met persona
class CreateWriterWithPersona(dspy.Module):
    """Ontdek verschillende perspectieven voor het onderzoeken van het onderwerp door Wikipedia-pagina's van gerelateerde onderwerpen te lezen."""

    def __init__(self, engine: Union[dspy.dsp.LM, dspy.dsp.HFModel]):
        super().__init__()
        self.find_related_topic = dspy.ChainOfThought(FindRelatedTopic)
        self.gen_persona = dspy.ChainOfThought(GenPersona)
        self.engine = engine

    def forward(self, topic: str, draft=None):
        with dspy.settings.context(lm=self.engine):
            # Haal sectienamen op van wiki-pagina's van relevante onderwerpen ter inspiratie
            related_topics = self.find_related_topic(topic=topic).related_topics
            urls = []
            for s in related_topics.split("\n"):
                if "http" in s:
                    urls.append(s[s.find("http"):])
            examples = []
            for url in urls:
                try:
                    title, toc = get_wiki_page_title_and_toc(url)
                    examples.append(f"Titel: {title}\nInhoudsopgave: {toc}")
                except Exception as e:
                    logging.error(f"Fout opgetreden bij het verwerken van {url}: {e}")
                    continue
            if len(examples) == 0:
                examples.append("N/A")
            gen_persona_output = self.gen_persona(
                topic=topic, examples="\n----------\n".join(examples)
            ).personas

        personas = []
        for s in gen_persona_output.split("\n"):
            match = re.search(r"\d+\.\s*(.*)", s)
            if match:
                personas.append(match.group(1))

        sorted_personas = personas

        return dspy.Prediction(
            personas=personas,
            raw_personas_output=sorted_personas,
            related_topics=related_topics,
        )

# Hoofdklasse voor het genereren van persona's
class StormPersonaGenerator:
    """
    Een generatorklasse voor het creëren van persona's op basis van een gegeven onderwerp.

    Deze klasse gebruikt een onderliggende engine om persona's te genereren die zijn afgestemd op het opgegeven onderwerp.
    De generator integreert met een `CreateWriterWithPersona`-instantie om diverse persona's te creëren,
    inclusief een standaard 'Basisfeiten schrijver' persona.

    Attributen:
        create_writer_with_persona (CreateWriterWithPersona): Een instantie verantwoordelijk voor
            het genereren van persona's op basis van de opgegeven engine en onderwerp.

    Args:
        engine (Union[dspy.dsp.LM, dspy.dsp.HFModel]): De onderliggende engine gebruikt voor het genereren
            van persona's. Het moet een instantie zijn van ofwel `dspy.dsp.LM` of `dspy.dsp.HFModel`.
    """

    def __init__(self, engine: Union[dspy.dsp.LM, dspy.dsp.HFModel]):
        self.create_writer_with_persona = CreateWriterWithPersona(engine=engine)

    def generate_persona(self, topic: str, max_num_persona: int = 3) -> List[str]:
        """
        Genereert een lijst van persona's op basis van het opgegeven onderwerp, tot een maximum aantal gespecificeerd.

        Deze methode creëert eerst persona's met behulp van de onderliggende `create_writer_with_persona`-instantie
        en voegt vervolgens een standaard 'Basisfeiten schrijver' persona toe aan het begin van de lijst voordat deze wordt geretourneerd.
        Het aantal geretourneerde persona's is beperkt tot `max_num_persona`, exclusief de standaardpersona.

        Args:
            topic (str): Het onderwerp waarvoor persona's moeten worden gegenereerd.
            max_num_persona (int): Het maximale aantal te genereren persona's, exclusief de
                standaard 'Basisfeiten schrijver' persona.

        Returns:
            List[str]: Een lijst met personabeschrijvingen, inclusief de standaard 'Basisfeiten schrijver' persona
                en tot `max_num_persona` extra persona's gegenereerd op basis van het onderwerp.
        """
        personas = self.create_writer_with_persona(topic=topic)
        default_persona = "Basisfeiten schrijver: Basisfeiten schrijver die zich richt op het breed behandelen van de basisfeiten over het onderwerp."
        considered_personas = [default_persona] + personas.personas[:max_num_persona]
        return considered_personas