# Agent QA Lab

A trace-aware regression harness for tool-using agents, with deterministic evaluation of tool-call correctness as its first quality gate.

Built on the [Strands Agents SDK](https://github.com/strands-agents/sdk-python).

## What it does

Agent QA Lab executes a Strands agent against a case pack, normalizes the execution traces into a stable format, evaluates tool-call correctness across five dimensions, and writes results to a JSON artifact.

### Evaluation dimensions

- **Correct tool selected** — did the agent call the expected tool?
- **Required arguments present** — were the necessary arguments included with correct values?
- **Forbidden tools avoided** — did the agent stay away from disallowed tools?
- **Unnecessary calls avoided** — did the agent avoid irrelevant tool calls?
- **Answer consistent with tool output** — does the final answer reflect what the tool actually returned?

## Project structure

```
agent-qa-lab/
  agents/
    sample_support_agent.py     # Strands agent factory (OpenAI-backed)
    normalizer.py               # Strands trace -> RunRecord normalization
  cases/
    support_cases.jsonl         # 12 test cases across 6 categories
  domain/
    fixtures.py                 # Canned tool response data
    support_tools.py            # @tool decorated functions
  evaluators/
    base.py                     # BaseEvaluator ABC
    tool_correctness.py         # ToolCorrectnessEvaluator (5 dimensions)
  runners/
    run_experiment.py           # CLI entry point
  schemas/
    case.py                     # TestCase, CaseExpectation
    run_record.py               # RunRecord, ToolCall, ConversationTurn
    results.py                  # EvaluationResult, ExperimentResult
  outputs/                      # JSON results (gitignored)
```

## Quickstart

### Prerequisites

- Python 3.10+
- An OpenAI API key

### Install

```bash
pip install -e .
```

### Run

```bash
export OPENAI_API_KEY="your-key-here"
python -m runners.run_experiment cases/support_cases.jsonl
```

Results are written to `outputs/`.

### Use a different model

```bash
export AQL_MODEL_ID="gpt-4o-mini"
python -m runners.run_experiment cases/support_cases.jsonl
```

## Example output

```
Loaded 12 cases from cases/support_cases.jsonl
[1/12] Running support_001 ...
  -> PASS
[2/12] Running support_002 ...
  -> PASS
...
[8/12] Running support_008 ...
  -> FAIL
     Required tool 'lookup_order_status' was not called
...
Results: 9/12 passed, 3 failed
Output:  outputs/support_phase0_20260319_222407.json
```

## Writing test cases

Cases are defined in JSONL format (one JSON object per line). Each case specifies:

```json
{
  "case_id": "support_001",
  "input": "What is the status of order ORD-1001?",
  "category": "happy_path",
  "expectation": {
    "required_tools": ["lookup_order_status"],
    "required_arguments": {"lookup_order_status": {"order_id": "ORD-1001"}},
    "forbidden_tools": ["refund_order"],
    "allowed_tools": ["lookup_order_status"],
    "expected_constraints": ["answer_consistent_with_tool_output"]
  }
}
```

### Expectation fields

| Field | Purpose |
|---|---|
| `required_tools` | Tools that must be called |
| `required_arguments` | Expected arguments per tool (tool_name -> {arg: value}) |
| `forbidden_tools` | Tools that must not be called |
| `allowed_tools` | Only these tools are acceptable (empty = no restriction) |
| `expected_constraints` | Behavioral constraints: `"no_tool_call_expected"`, `"answer_consistent_with_tool_output"` |

### Included case categories

The bundled case pack (`cases/support_cases.jsonl`) includes 12 cases across 6 categories:

| Category | What it tests |
|---|---|
| `happy_path` | Agent calls the right tool with the right args and answers correctly |
| `missing_identifier` | Agent should ask for clarification, not guess |
| `wrong_tool_temptation` | Agent is tempted toward an irrelevant tool |
| `forbidden_behavior` | Agent must not take a disallowed action |
| `unnecessary_tool_call` | Agent should not call tools when none are needed |
| `output_consistency` | Agent's answer must match what the tool returned |

## Reading the output

The JSON artifact in `outputs/` contains:

```json
{
  "experiment_name": "support_phase0",
  "timestamp": "2026-03-19T22:24:07+00:00",
  "summary": {
    "total_cases": 12,
    "passed_cases": 9,
    "failed_cases": 3
  },
  "results": [
    {
      "case_id": "support_001",
      "run_record": {
        "tool_calls": [...],
        "final_output": "Order ORD-1001 has been shipped..."
      },
      "evaluation": {
        "passed": true,
        "dimension_results": {
          "correct_tool_selected": true,
          "required_arguments_present": true,
          "forbidden_tools_avoided": true,
          "unnecessary_calls_avoided": true,
          "final_answer_consistent_with_tool_output": true
        },
        "notes": []
      }
    }
  ]
}
```

Failed cases include `notes` explaining what went wrong, e.g.:
```
"notes": ["Required tool 'lookup_order_status' was not called"]
```

## Adding a custom evaluator

Subclass `BaseEvaluator` and implement `evaluate`:

```python
from evaluators.base import BaseEvaluator
from schemas.case import TestCase
from schemas.run_record import RunRecord
from schemas.results import EvaluationResult

class MyEvaluator(BaseEvaluator):
    name = "my_evaluator"

    def evaluate(self, case: TestCase, run_record: RunRecord) -> EvaluationResult:
        passed = "some_keyword" in run_record.final_output.lower()
        return EvaluationResult(
            case_id=case.case_id,
            evaluator_name=self.name,
            passed=passed,
            dimension_results={"keyword_present": passed},
            notes=[] if passed else ["Expected keyword not found"],
        )
```

## Architecture

The key architectural decision is the **normalized RunRecord boundary**. Strands-specific code (agent execution, trace extraction) lives in `agents/`. Everything downstream — evaluators, result schemas, JSON output — is framework-agnostic and operates only on `RunRecord` objects.

```
Strands Agent -> raw trace -> StrandsTraceNormalizer -> RunRecord -> Evaluator -> JSON
```

This means you can swap the agent framework without touching the evaluator or reporting layers.

## License

MIT
