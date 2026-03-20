from typing import Any

from pydantic import BaseModel

from schemas.run_record import RunRecord


class EvaluationResult(BaseModel):
    case_id: str
    evaluator_name: str
    passed: bool
    dimension_results: dict[str, bool]
    notes: list[str] = []
    raw_details: dict[str, Any] = {}


class CaseResult(BaseModel):
    case_id: str
    run_record: RunRecord
    evaluation: EvaluationResult


class ExperimentSummary(BaseModel):
    total_cases: int
    passed_cases: int
    failed_cases: int


class ExperimentResult(BaseModel):
    experiment_name: str
    timestamp: str
    summary: ExperimentSummary
    results: list[CaseResult]
