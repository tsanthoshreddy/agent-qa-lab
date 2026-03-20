"""Run an experiment: execute a Strands agent against a case pack and evaluate.

Usage:
    python -m runners.run_experiment cases/support_cases.jsonl
    python -m runners.run_experiment cases/support_cases.jsonl --output-dir outputs/
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from agents.normalizer import StrandsTraceNormalizer
from agents.sample_support_agent import create_support_agent
from evaluators.tool_correctness import ToolCorrectnessEvaluator
from schemas.case import TestCase
from schemas.results import CaseResult, ExperimentResult, ExperimentSummary
from schemas.run_record import RunRecord


def load_cases(path: Path) -> list[TestCase]:
    cases = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            cases.append(TestCase.model_validate(json.loads(line)))
    return cases


def run_single_case(
    normalizer: StrandsTraceNormalizer,
    evaluator: ToolCorrectnessEvaluator,
    case: TestCase,
) -> CaseResult:
    # Create a fresh agent per case for full isolation
    agent = create_support_agent()

    try:
        agent(case.input)
        run_record = normalizer.normalize(
            messages=agent.messages,
            case_id=case.case_id,
            input_text=case.input,
        )
    except Exception as exc:
        run_record = RunRecord(
            run_id=f"run_error_{case.case_id}",
            case_id=case.case_id,
            agent_name=normalizer.agent_name,
            input_text=case.input,
            conversation_turns=[],
            tool_calls=[],
            final_output=f"AGENT ERROR: {exc}",
            metadata={"error": str(exc)},
        )

    evaluation = evaluator.evaluate(case, run_record)
    return CaseResult(
        case_id=case.case_id, run_record=run_record, evaluation=evaluation
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Agent QA Lab experiment")
    parser.add_argument("cases_file", type=Path, help="Path to JSONL case pack")
    parser.add_argument(
        "--output-dir", type=Path, default=Path("outputs"), help="Output directory"
    )
    parser.add_argument(
        "--experiment-name", default="support_phase0", help="Experiment name"
    )
    args = parser.parse_args()

    cases = load_cases(args.cases_file)
    print(f"Loaded {len(cases)} cases from {args.cases_file}", file=sys.stderr)

    normalizer = StrandsTraceNormalizer()
    evaluator = ToolCorrectnessEvaluator()

    results: list[CaseResult] = []
    for i, case in enumerate(cases, 1):
        print(f"[{i}/{len(cases)}] Running {case.case_id} ...", file=sys.stderr)
        case_result = run_single_case(normalizer, evaluator, case)
        status = "PASS" if case_result.evaluation.passed else "FAIL"
        print(f"  -> {status}", file=sys.stderr)
        if case_result.evaluation.notes:
            for note in case_result.evaluation.notes:
                print(f"     {note}", file=sys.stderr)
        results.append(case_result)

    passed = sum(1 for r in results if r.evaluation.passed)
    experiment = ExperimentResult(
        experiment_name=args.experiment_name,
        timestamp=datetime.now(timezone.utc).isoformat(),
        summary=ExperimentSummary(
            total_cases=len(results),
            passed_cases=passed,
            failed_cases=len(results) - passed,
        ),
        results=results,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = args.output_dir / f"{args.experiment_name}_{ts}.json"
    out_path.write_text(experiment.model_dump_json(indent=2))

    print(f"\n{'='*50}", file=sys.stderr)
    print(
        f"Results: {passed}/{len(results)} passed, "
        f"{len(results) - passed} failed",
        file=sys.stderr,
    )
    print(f"Output:  {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
