from evaluators.tool_correctness import ToolCorrectnessEvaluator
from schemas.case import CaseExpectation, TestCase
from schemas.run_record import RunRecord, ToolCall


def _make_case(
    case_id: str = "test_001",
    input_text: str = "Check order ORD-1001",
    **expectation_kwargs,
) -> TestCase:
    return TestCase(
        case_id=case_id,
        input=input_text,
        category="test",
        expectation=CaseExpectation(**expectation_kwargs),
    )


def _make_run(
    case_id: str = "test_001",
    tool_calls: list[ToolCall] | None = None,
    final_output: str = "Your order has shipped.",
) -> RunRecord:
    return RunRecord(
        run_id="run_test",
        case_id=case_id,
        agent_name="test_agent",
        input_text="test input",
        conversation_turns=[],
        tool_calls=tool_calls or [],
        final_output=final_output,
    )


evaluator = ToolCorrectnessEvaluator()


class TestHappyPath:
    def test_all_pass(self):
        case = _make_case(
            required_tools=["lookup_order_status"],
            required_arguments={"lookup_order_status": {"order_id": "ORD-1001"}},
            forbidden_tools=["refund_order"],
            allowed_tools=["lookup_order_status"],
            expected_constraints=["answer_consistent_with_tool_output"],
            answer_must_contain=["shipped"],
        )
        run = _make_run(
            tool_calls=[
                ToolCall(
                    tool_name="lookup_order_status",
                    arguments={"order_id": "ORD-1001"},
                    output={"status": "shipped"},
                    call_index=0,
                )
            ],
            final_output="Your order ORD-1001 has shipped.",
        )
        result = evaluator.evaluate(case, run)
        assert result.passed is True
        assert all(result.dimension_results.values())


class TestCorrectToolSelected:
    def test_required_tool_not_called(self):
        case = _make_case(required_tools=["lookup_order_status"])
        run = _make_run(tool_calls=[])
        result = evaluator.evaluate(case, run)
        assert result.dimension_results["correct_tool_selected"] is False
        assert result.passed is False

    def test_no_tool_expected_but_tool_called(self):
        case = _make_case(expected_constraints=["no_tool_call_expected"])
        run = _make_run(
            tool_calls=[
                ToolCall(
                    tool_name="lookup_order_status",
                    arguments={},
                    call_index=0,
                )
            ]
        )
        result = evaluator.evaluate(case, run)
        assert result.dimension_results["correct_tool_selected"] is False

    def test_no_tool_expected_and_none_called(self):
        case = _make_case(expected_constraints=["no_tool_call_expected"])
        run = _make_run(tool_calls=[])
        result = evaluator.evaluate(case, run)
        assert result.passed is True


class TestRequiredArguments:
    def test_wrong_arguments(self):
        case = _make_case(
            required_tools=["lookup_order_status"],
            required_arguments={"lookup_order_status": {"order_id": "ORD-1001"}},
        )
        run = _make_run(
            tool_calls=[
                ToolCall(
                    tool_name="lookup_order_status",
                    arguments={"order_id": "ORD-9999"},
                    call_index=0,
                )
            ]
        )
        result = evaluator.evaluate(case, run)
        assert result.dimension_results["required_arguments_present"] is False

    def test_multiple_calls_one_matches(self):
        case = _make_case(
            required_tools=["lookup_order_status"],
            required_arguments={"lookup_order_status": {"order_id": "ORD-1001"}},
        )
        run = _make_run(
            tool_calls=[
                ToolCall(
                    tool_name="lookup_order_status",
                    arguments={"order_id": "ORD-9999"},
                    call_index=0,
                ),
                ToolCall(
                    tool_name="lookup_order_status",
                    arguments={"order_id": "ORD-1001"},
                    call_index=1,
                ),
            ]
        )
        result = evaluator.evaluate(case, run)
        assert result.dimension_results["required_arguments_present"] is True


class TestForbiddenTools:
    def test_forbidden_tool_called(self):
        case = _make_case(
            required_tools=["lookup_order_status"],
            forbidden_tools=["refund_order"],
        )
        run = _make_run(
            tool_calls=[
                ToolCall(
                    tool_name="lookup_order_status",
                    arguments={},
                    call_index=0,
                ),
                ToolCall(
                    tool_name="refund_order",
                    arguments={"order_id": "ORD-1001", "reason": "test"},
                    call_index=1,
                ),
            ]
        )
        result = evaluator.evaluate(case, run)
        assert result.dimension_results["forbidden_tools_avoided"] is False
        assert result.passed is False

    def test_no_forbidden_tools_defined(self):
        case = _make_case(required_tools=["lookup_order_status"])
        run = _make_run(
            tool_calls=[
                ToolCall(
                    tool_name="lookup_order_status",
                    arguments={},
                    call_index=0,
                ),
            ]
        )
        result = evaluator.evaluate(case, run)
        assert result.dimension_results["forbidden_tools_avoided"] is True


class TestUnnecessaryCalls:
    def test_tool_outside_allowed_set(self):
        case = _make_case(
            required_tools=["lookup_order_status"],
            allowed_tools=["lookup_order_status"],
        )
        run = _make_run(
            tool_calls=[
                ToolCall(
                    tool_name="lookup_order_status",
                    arguments={},
                    call_index=0,
                ),
                ToolCall(
                    tool_name="lookup_customer_info",
                    arguments={},
                    call_index=1,
                ),
            ]
        )
        result = evaluator.evaluate(case, run)
        assert result.dimension_results["unnecessary_calls_avoided"] is False
        assert result.passed is False


class TestAnswerConsistency:
    def test_must_contain_pass(self):
        case = _make_case(
            expected_constraints=["answer_consistent_with_tool_output"],
            answer_must_contain=["shipped"],
        )
        run = _make_run(final_output="Your order has shipped.")
        result = evaluator.evaluate(case, run)
        assert (
            result.dimension_results["final_answer_consistent_with_tool_output"] is True
        )

    def test_must_contain_fail(self):
        case = _make_case(
            expected_constraints=["answer_consistent_with_tool_output"],
            answer_must_contain=["shipped"],
        )
        run = _make_run(final_output="Your order is delayed.")
        result = evaluator.evaluate(case, run)
        assert (
            result.dimension_results["final_answer_consistent_with_tool_output"]
            is False
        )

    def test_must_not_contain_fail(self):
        case = _make_case(
            expected_constraints=["answer_consistent_with_tool_output"],
            answer_must_not_contain=["delayed"],
        )
        run = _make_run(final_output="Your order is delayed.")
        result = evaluator.evaluate(case, run)
        assert (
            result.dimension_results["final_answer_consistent_with_tool_output"]
            is False
        )

    def test_must_contain_case_insensitive(self):
        case = _make_case(
            expected_constraints=["answer_consistent_with_tool_output"],
            answer_must_contain=["Shipped"],
        )
        run = _make_run(final_output="your order has SHIPPED successfully")
        result = evaluator.evaluate(case, run)
        assert (
            result.dimension_results["final_answer_consistent_with_tool_output"] is True
        )

    def test_no_constraint_skips_check(self):
        case = _make_case(expected_constraints=[])
        run = _make_run(final_output="Anything goes here.")
        result = evaluator.evaluate(case, run)
        assert (
            result.dimension_results["final_answer_consistent_with_tool_output"] is True
        )

    def test_fallback_heuristic_skips_short_values(self):
        """The heuristic fallback should skip values shorter than 4 chars."""
        case = _make_case(
            required_tools=["lookup_order_status"],
            expected_constraints=["answer_consistent_with_tool_output"],
        )
        run = _make_run(
            tool_calls=[
                ToolCall(
                    tool_name="lookup_order_status",
                    arguments={"order_id": "ORD-1001"},
                    output={"status": "ok", "id": "1"},
                    call_index=0,
                )
            ],
            final_output="Everything is fine with your order.",
        )
        result = evaluator.evaluate(case, run)
        # "ok" and "1" are < 4 chars, so skipped; no usable values -> passes
        assert (
            result.dimension_results["final_answer_consistent_with_tool_output"] is True
        )


class TestAllDimensionsCritical:
    def test_forbidden_tool_fails_overall(self):
        """Even if other dimensions pass, forbidden tool should fail the run."""
        case = _make_case(
            required_tools=["lookup_order_status"],
            forbidden_tools=["refund_order"],
            expected_constraints=["answer_consistent_with_tool_output"],
            answer_must_contain=["shipped"],
        )
        run = _make_run(
            tool_calls=[
                ToolCall(
                    tool_name="lookup_order_status",
                    arguments={},
                    call_index=0,
                ),
                ToolCall(
                    tool_name="refund_order",
                    arguments={"order_id": "ORD-1001", "reason": "x"},
                    call_index=1,
                ),
            ],
            final_output="Your order has shipped. Refund initiated.",
        )
        result = evaluator.evaluate(case, run)
        assert result.dimension_results["forbidden_tools_avoided"] is False
        assert result.passed is False
