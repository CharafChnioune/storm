import logging
import re
from typing import Union, List

import dspy
import requests
from bs4 import BeautifulSoup

# Configureer logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def get_wiki_page_title_and_toc(url):
    """Haal de hoofdtitel en inhoudsopgave op van een Wikipedia-pagina URL."""
    logger.info(f"Ophalen van titel en inhoudsopgave voor URL: {url}")
    try:
        response = requests.get(url)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')

        main_title = soup.find('h1').text.replace('[edit]', '').strip().replace('\xa0', ' ')
        logger.info(f"Hoofdtitel opgehaald: {main_title}")

        toc = ""
        levels = []
        excluded_sections = {'Contents', 'See also', 'Notes', 'References', 'External links'}

        for header in soup.find_all(['h2', 'h3', "h4", "h5", "h6"]):
            level = int(header.name[1])
            section_title = header.text.replace('[edit]', '').strip().replace('\xa0', ' ')
            if section_title in excluded_sections:
                continue

            while levels and level <= levels[-1]:
                levels.pop()
            levels.append(level)

            indentation = "  " * (len(levels) - 1)
            toc += f"{indentation}{section_title}\n"

        logger.info("Inhoudsopgave succesvol gegenereerd")
        return main_title, toc.strip()
    except Exception as e:
        logger.error(f"Fout bij het ophalen van titel en inhoudsopgave: {str(e)}")
        raise

class FindRelatedTopic(dspy.Signature):
    """Identificeer en aanbeveel enkele Wikipedia-pagina's over nauw verwante onderwerpen."""

    topic = dspy.InputField(prefix='Interessegebied:', format=str)
    related_topics = dspy.OutputField(format=str)

class GenPersona(dspy.Signature):
    """Selecteer een groep Wikipedia-editors die samenwerken aan een uitgebreid artikel over het onderwerp."""

    topic = dspy.InputField(prefix='Interessegebied:', format=str)
    examples = dspy.InputField(prefix='Wiki-pagina outlines van verwante onderwerpen ter inspiratie:\n', format=str)
    personas = dspy.OutputField(format=str)

class CreateWriterWithPersona(dspy.Module):
    """Ontdek verschillende perspectieven voor het onderzoeken van het onderwerp door Wikipedia-pagina's van verwante onderwerpen te lezen."""

    def __init__(self, engine: Union[dspy.dsp.LM, dspy.dsp.HFModel]):
        super().__init__()
        self.find_related_topic = dspy.ChainOfThought(FindRelatedTopic)
        self.gen_persona = dspy.ChainOfThought(GenPersona)
        self.engine = engine
        logger.info("CreateWriterWithPersona module geïnitialiseerd")

    def forward(self, topic: str, draft=None):
        logger.info(f"Start persona creatie voor onderwerp: {topic}")
        with dspy.settings.context(lm=self.engine):
            related_topics = self.find_related_topic(topic=topic).related_topics
            logger.info(f"Gerelateerde onderwerpen gevonden: {related_topics}")

            urls = [s[s.find('http'):] for s in related_topics.split('\n') if 'http' in s]
            examples = []
            for url in urls:
                try:
                    title, toc = get_wiki_page_title_and_toc(url)
                    examples.append(f'Title: {title}\nTable of Contents: {toc}')
                    logger.info(f"Voorbeeld toegevoegd voor URL: {url}")
                except Exception as e:
                    logger.error(f'Fout bij het verwerken van {url}: {e}')
                    continue
            
            if len(examples) == 0:
                logger.warning("Geen voorbeelden gevonden, gebruikt 'N/A'")
                examples.append('N/A')
            
            gen_persona_output = self.gen_persona(topic=topic, examples='\n----------\n'.join(examples)).personas
            logger.info("Persona's gegenereerd")

        personas = [match.group(1) for match in re.finditer(r'\d+\.\s*(.*)', gen_persona_output)]
        logger.info(f"Aantal gegenereerde persona's: {len(personas)}")

        return dspy.Prediction(personas=personas, raw_personas_output=personas, related_topics=related_topics)

class StormPersonaGenerator():
    """Een generator klasse voor het creëren van persona's op basis van een gegeven onderwerp."""

    def __init__(self, engine: Union[dspy.dsp.LM, dspy.dsp.HFModel]):
        self.create_writer_with_persona = CreateWriterWithPersona(engine=engine)
        logger.info("StormPersonaGenerator geïnitialiseerd")

    def generate_persona(self, topic: str, max_num_persona: int = 3) -> List[str]:
        """Genereert een lijst van persona's op basis van het opgegeven onderwerp."""
        logger.info(f"Start persona generatie voor onderwerp: {topic}, max aantal: {max_num_persona}")
        personas = self.create_writer_with_persona(topic=topic)
        default_persona = 'Basic fact writer: Basic fact writer focusing on broadly covering the basic facts about the topic.'
        considered_personas = [default_persona] + personas.personas[:max_num_persona]
        logger.info(f"Aantal gegenereerde persona's (inclusief default): {len(considered_personas)}")
        return considered_personas

# Voeg een handler toe om logs naar een bestand te schrijven
file_handler = logging.FileHandler('storm_persona_generation.log')
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logger.addHandler(file_handler)

logger.info("Script geïnitialiseerd en logging geconfigureerd")