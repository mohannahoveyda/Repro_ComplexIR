from abc import ABC, abstractmethod
from typing import Any, Dict

class ParserABC(ABC):
    @abstractmethod
    def parse_query(self, query: str) -> Dict[str, Any]:
        """
        Convert natural‐language query into a dict
        containing at least 'atoms': List[str]
        and        'logical query': str
        """