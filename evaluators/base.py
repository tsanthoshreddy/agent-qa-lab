from abc import ABC, abstractmethod

from schemas.case import EvalCase
from schemas.results import EvaluationResult
from schemas.run_record import RunRecord


class BaseEvaluator(ABC):
    name: str

    @abstractmethod
    def evaluate(self, case: EvalCase, run_record: RunRecord) -> EvaluationResult:
        ...
