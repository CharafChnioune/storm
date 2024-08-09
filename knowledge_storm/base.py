from abc import ABC, abstractmethod
from typing import Union, List, Dict

class Retriever(ABC):
    @abstractmethod
    def retrieve(self, query: Union[str, List[str]], exclude_urls: List[str] = []) -> List[Union[Dict, 'StormInformation']]:
        pass

class StormInformation:
    def __init__(self, snippets, score, url, title):
        self.snippets = snippets
        self.score = score
        self.url = url
        self.title = title

    @classmethod
    def from_dict(cls, data):
        return cls(
            snippets=data.get('snippets', []),
            score=data.get('score', 0),
            url=data.get('url', ''),
            title=data.get('title', '')
        )