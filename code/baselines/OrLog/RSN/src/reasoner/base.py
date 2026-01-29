# src/reasoner/base.py
# from abc import ABC, abstractmethod
# from typing import List, Dict, Tuple

# class ReasonerABC(ABC):
#     @abstractmethod
#     def rank_entities(
#         self,
#         query: str,
#         candidates: List[Dict]  # your entry["pred_docs_metadata"]
#     ) -> List[Tuple[str, float]]:
#         """
#         Given the raw query string and each entity’s metadata
#         (doc id, wiki text, props, etc.), return a sorted list
#         of (entity_id, score).
#         """

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Tuple

class ReasonerABC(ABC):
    @abstractmethod
    def rank_entities(
        self,
        entry: Dict[str, Any]
    ) -> List[Tuple[str, float]]:
        """
        Given one “entry” from the dataset (with keys like
        'query', 'pred_docs_metadata', etc.),
        return a list of (doc_id, score), sorted descending.
        """