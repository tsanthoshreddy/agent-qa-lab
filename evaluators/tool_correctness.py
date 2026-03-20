from schemas.case import EvalCase
from schemas.results import EvaluationResult
from schemas.run_record import RunRecord

from evaluators.base import BaseEvaluator


class ToolCorrectnessEvaluator(BaseEvaluator):
    name = "tool_correctness"

    def evaluate(self, case: EvalCase, run_record: RunRecord) -> EvaluationResult:
        dimensions: dict[str, bool] = {}
        notes: list[str] = []

        dimensions["correct_tool_selected"] = self._check_correct_tool(
            case, run_record, notes
        )
        dimensions["required_arguments_present"] = self._check_required_arguments(
            case, run_record, notes
        )
        dimensions["forbidden_tools_avoided"] = self._check_forbidden_tools(
            case, run_record, notes
        )
        dimensions["unnecessary_calls_avoided"] = self._check_unnecessary_calls(
            case, run_record, notes
        )
        dimensions["final_answer_consistent_with_tool_output"] = (
            self._check_answer_consistency(case, run_record, notes)
        )

        passed = all(dimensions.values())

        return EvaluationResult(
            case_id=case.case_id,
            evaluator_name=self.name,
            passed=passed,
            dimension_results=dimensions,
            notes=notes,
        )

    @staticmethod
    def _check_correct_tool(
        case: EvalCase, run_record: RunRecord, notes: list[str]
    ) -> bool:
        exp = case.expectation
        called_tools = {tc.tool_name for tc in run_record.tool_calls}

        # If no tool call is expected, verify none were made
        if "no_tool_call_expected" in exp.expected_constraints:
            if called_tools:
                notes.append(
                    f"Expected no tool calls but got: {sorted(called_tools)}"
                )
                return False
            return True

        # Check all required tools were called
        for tool_name in exp.required_tools:
            if tool_name not in called_tools:
                notes.append(f"Required tool '{tool_name}' was not called")
                return False

        return True

    @staticmethod
    def _check_required_arguments(
        case: EvalCase, run_record: RunRecord, notes: list[str]
    ) -> bool:
        exp = case.expectation
        if not exp.required_arguments:
            return True

        for tool_name, expected_args in exp.required_arguments.items():
            matching_calls = [
                tc for tc in run_record.tool_calls if tc.tool_name == tool_name
            ]
            if not matching_calls:
                notes.append(
                    f"Cannot check arguments for '{tool_name}': tool was not called"
                )
                return False

            # Check at least one matching call has the right arguments
            found = False
            for tc in matching_calls:
                if all(
                    str(tc.arguments.get(k)) == str(v)
                    for k, v in expected_args.items()
                ):
                    found = True
                    break

            if not found:
                actual_args = [tc.arguments for tc in matching_calls]
                notes.append(
                    f"Tool '{tool_name}' called but arguments don't match. "
                    f"Expected {expected_args}, got {actual_args}"
                )
                return False

        return True

    @staticmethod
    def _check_forbidden_tools(
        case: EvalCase, run_record: RunRecord, notes: list[str]
    ) -> bool:
        exp = case.expectation
        if not exp.forbidden_tools:
            return True

        called_tools = {tc.tool_name for tc in run_record.tool_calls}
        violations = called_tools & set(exp.forbidden_tools)
        if violations:
            notes.append(f"Forbidden tool(s) were called: {sorted(violations)}")
            return False

        return True

    @staticmethod
    def _check_unnecessary_calls(
        case: EvalCase, run_record: RunRecord, notes: list[str]
    ) -> bool:
        exp = case.expectation

        # If no tool call expected, any call is unnecessary
        if "no_tool_call_expected" in exp.expected_constraints:
            if run_record.tool_calls:
                names = [tc.tool_name for tc in run_record.tool_calls]
                notes.append(f"Unnecessary tool calls made: {names}")
                return False
            return True

        # If allowed_tools is set, every call must be in the list
        if exp.allowed_tools:
            called_tools = {tc.tool_name for tc in run_record.tool_calls}
            disallowed = called_tools - set(exp.allowed_tools)
            if disallowed:
                notes.append(
                    f"Tool calls outside allowed set: {sorted(disallowed)}"
                )
                return False

        return True

    @staticmethod
    def _check_answer_consistency(
        case: EvalCase, run_record: RunRecord, notes: list[str]
    ) -> bool:
        exp = case.expectation

        if "answer_consistent_with_tool_output" not in exp.expected_constraints:
            return True

        answer_lower = run_record.final_output.lower()
        ok = True

        # Check explicit must-contain values
        for value in exp.answer_must_contain:
            if value.lower() not in answer_lower:
                notes.append(
                    f"Answer must contain '{value}' but it was not found"
                )
                ok = False

        # Check explicit must-not-contain values
        for value in exp.answer_must_not_contain:
            if value.lower() in answer_lower:
                notes.append(
                    f"Answer must not contain '{value}' but it was found"
                )
                ok = False

        # If explicit assertions were defined, use them as the verdict
        if exp.answer_must_contain or exp.answer_must_not_contain:
            return ok

        # Fallback: heuristic check against tool output values
        if not run_record.tool_calls:
            notes.append(
                "Answer consistency check skipped: no tool calls to compare against"
            )
            return True

        check_values: list[str] = []
        for tc in run_record.tool_calls:
            if isinstance(tc.output, dict):
                for v in tc.output.values():
                    sv = str(v)
                    # Skip short/generic values to avoid false positives
                    if v is None or len(sv) < 4:
                        continue
                    check_values.append(sv)
            elif isinstance(tc.output, str) and len(tc.output) >= 4:
                check_values.append(tc.output)

        if not check_values:
            notes.append("Answer consistency check skipped: no usable tool output")
            return True

        matched_any = any(v.lower() in answer_lower for v in check_values)

        if not matched_any:
            notes.append(
                f"Final answer may not reflect tool output. "
                f"Checked for any of {check_values} in the answer."
            )
            return False

        return True
