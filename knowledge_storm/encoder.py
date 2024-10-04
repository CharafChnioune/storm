import requests
import os
from typing import List, Tuple, Union, Optional, Dict, Literal
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
import dspy

# Basisklasse voor embedding modellen
class EmbeddingModel:
    def __init__(self):
        pass

    def get_embedding(self, text: str) -> Tuple[np.ndarray, int]:
        raise Exception("Niet geïmplementeerd")

# OpenAI embedding model implementatie
class OpenAIEmbeddingModel(EmbeddingModel):
    def __init__(self, model: str = "text-embedding-3-small", api_key: str = None):
        # Gebruik omgevingsvariabele als geen API-sleutel is opgegeven
        if not api_key:
            api_key = os.getenv("OPENAI_API_KEY")

        self.url = "https://api.openai.com/v1/embeddings"
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        }
        self.model = model

    def get_embedding(self, text: str) -> Tuple[np.ndarray, int]:
        data = {"input": text, "model": self.model}

        # API-verzoek verzenden en response verwerken
        response = requests.post(self.url, headers=self.headers, json=data)
        if response.status_code == 200:
            data = response.json()
            embedding = np.array(data["data"][0]["embedding"])
            token = data["usage"]["prompt_tokens"]
            return embedding, token
        else:
            response.raise_for_status()

# Together AI embedding model implementatie
class TogetherEmbeddingModel:
    def __init__(self, model: str = "BAAI/bge-large-en-v1.5", api_key: str = None):
        import together

        self.model = model
        if not api_key:
            api_key = os.getenv("TOGETHER_API_KEY")
        self.together_client = together.Together(api_key=api_key)

    def get_embedding(self, text: str) -> Tuple[np.ndarray, int]:
        response = self.together_client.embeddings.create(input=text, model=self.model)
        # Opmerking: Together API geeft geen tokengebruik terug, dus we retourneren -1
        return response.data[0].embedding, -1

# Azure OpenAI embedding model implementatie
class AzureOpenAIEmbeddingModel:
    def __init__(self, model: str = "text-embedding-3-small", api_key: str = None):
        from openai import AzureOpenAI

        self.model = model
        if not api_key:
            api_key = os.getenv("AZURE_API_KEY")

        self.client = AzureOpenAI(
            api_key=api_key,
            api_version=os.getenv("AZURE_API_VERSION"),
            azure_endpoint=os.getenv("AZURE_API_BASE"),
        )

    def get_embedding(self, text: str) -> Tuple[np.ndarray, int]:
        response = self.client.embeddings.create(input=text, model=self.model)

        embedding = np.array(response.data[0].embedding)
        token = response.usage.prompt_tokens
        return embedding, token

# Ollama embedding model implementatie
class OllamaEmbeddingModel(EmbeddingModel):
    def __init__(self, model: str = "charaf/bge-m3-f32:latest", port: int = 11434, url: str = "http://localhost"):
        self.model = model
        self.port = port
        self.url = url
        self.client = dspy.OllamaLocal(model=model, base_url=f"{url}:{port}")

    def get_embedding(self, text: str) -> Tuple[np.ndarray, int]:
        response = self.client.embed(text)
        embedding = np.array(response['embedding'])
        # Ollama geeft geen tokengebruik terug, dus we schatten het op basis van de tekstlengte
        estimated_tokens = len(text.split())
        return embedding, estimated_tokens

# Hoofdfunctie voor het ophalen van tekstembeddings
def get_text_embeddings(
    texts: Union[str, List[str]],
    max_workers: int = 5,
    embedding_cache: Optional[Dict[str, np.ndarray]] = None,
) -> Tuple[np.ndarray, int]:
    """
    Haal tekstembeddings op met behulp van het gespecificeerde embedding model.

    Args:
        texts (Union[str, List[str]]): Een enkele tekst of een lijst van teksten om te embedden.
        max_workers (int): Het maximale aantal workers voor parallelle verwerking.
        embedding_cache (Optional[Dict[str, np.ndarray]]): Een cache om eerder berekende embeddings op te slaan.

    Returns:
        Tuple[np.ndarray, int]: De 2D array van embeddings en het totale tokengebruik.
    """
    # Bepaal het te gebruiken embedding model op basis van de omgevingsvariabele
    embedding_model = None
    encoder_type = os.getenv("ENCODER_API_TYPE")
    if encoder_type and encoder_type == "openai":
        embedding_model = OpenAIEmbeddingModel()
    elif encoder_type and encoder_type == "azure":
        embedding_model = AzureOpenAIEmbeddingModel()
    elif encoder_type == "together":
        embedding_model = TogetherEmbeddingModel()
    elif encoder_type == "ollama":
        embedding_model = OllamaEmbeddingModel()
    else:
        raise Exception(
            "Geen geldig encodertype opgegeven. Controleer <repo root>/secrets.toml voor het veld ENCODER_API_TYPE"
        )

    def fetch_embedding(text: str) -> Tuple[str, np.ndarray, int]:
        # Gebruik cache indien beschikbaar
        if embedding_cache is not None and text in embedding_cache:
            return (
                text,
                embedding_cache[text],
                0,
            )  # Retourneer 0 tokens omdat er geen API-aanroep wordt gedaan
        embedding, token_usage = embedding_model.get_embedding(text)
        return text, embedding, token_usage

    # Verwerk enkele tekst of lijst van teksten
    if isinstance(texts, str):
        _, embedding, tokens = fetch_embedding(texts)
        return np.array(embedding), tokens

    embeddings = []
    total_tokens = 0

    # Parallelle verwerking van meerdere teksten
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(fetch_embedding, text): text for text in texts}

        for future in as_completed(futures):
            try:
                text, embedding, tokens = future.result()
                embeddings.append((text, embedding, tokens))
                total_tokens += tokens
            except Exception as e:
                print(f"Er is een fout opgetreden voor tekst: {futures[future]}")
                print(e)

    # Sorteer resultaten om de volgorde van de invoerteksten te behouden
    embeddings.sort(key=lambda x: texts.index(x[0]))
    if embedding_cache is not None:
        for text, embedding, _ in embeddings:
            embedding_cache[text] = embedding
    embeddings = [result[1] for result in embeddings]

    return np.array(embeddings), total_tokens