# src/estimator/base.py

from abc import ABC, abstractmethod
from typing import List, Dict, Any

class ProbabilityEstimatorABC(ABC):

    @abstractmethod
    def get_probabilities(
        self, 
        atoms: List[str], 
        entity_meta: Dict[str, Any]
    ) -> Dict[str, float]:
        """
        Given a list of atomic statements and an entity’s metadata,
        return a mapping atom → P(True).
        """