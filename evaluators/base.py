from abc import ABC, abstractmethod

from schemas.case import TestCase
from schemas.results import EvaluationResult
from schemas.run_record import RunRecord


class BaseEvaluator(ABC):
    name: str

    @abstractmethod
    def evaluate(self, case: TestCase, run_record: RunRecord) -> EvaluationResult:
        ...
