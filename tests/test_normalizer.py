from agents.normalizer import StrandsTraceNormalizer

normalizer = StrandsTraceNormalizer(agent_name="test_agent")


class TestBasicNormalization:
    def test_single_tool_call(self):
        messages = [
            {"role": "user", "content": [{"text": "Check order ORD-1001"}]},
            {
                "role": "assistant",
                "content": [
                    {
                        "toolUse": {
                            "toolUseId": "tu_1",
                            "name": "lookup_order_status",
                            "input": {"order_id": "ORD-1001"},
                        }
                    }
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "toolResult": {
                            "toolUseId": "tu_1",
                            "status": "success",
                            "content": [
                                {"text": '{"status": "shipped", "eta": "2026-03-22"}'}
                            ],
                        }
                    }
                ],
            },
            {
                "role": "assistant",
                "content": [{"text": "Order ORD-1001 has shipped, ETA 2026-03-22."}],
            },
        ]
        record = normalizer.normalize(messages, "case_001", "Check order ORD-1001")
        assert record.case_id == "case_001"
        assert len(record.tool_calls) == 1
        assert record.tool_calls[0].tool_name == "lookup_order_status"
        assert record.tool_calls[0].arguments == {"order_id": "ORD-1001"}
        assert record.tool_calls[0].output == {"status": "shipped", "eta": "2026-03-22"}
        assert record.tool_calls[0].call_index == 0
        assert record.final_output == "Order ORD-1001 has shipped, ETA 2026-03-22."

    def test_no_tool_calls(self):
        messages = [
            {"role": "user", "content": [{"text": "Hello"}]},
            {"role": "assistant", "content": [{"text": "Hi there!"}]},
        ]
        record = normalizer.normalize(messages, "case_002", "Hello")
        assert len(record.tool_calls) == 0
        assert record.final_output == "Hi there!"

    def test_multiple_tool_calls(self):
        messages = [
            {"role": "user", "content": [{"text": "Check two orders"}]},
            {
                "role": "assistant",
                "content": [
                    {
                        "toolUse": {
                            "toolUseId": "tu_1",
                            "name": "lookup_order_status",
                            "input": {"order_id": "ORD-1001"},
                        }
                    },
                    {
                        "toolUse": {
                            "toolUseId": "tu_2",
                            "name": "lookup_order_status",
                            "input": {"order_id": "ORD-1002"},
                        }
                    },
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "toolResult": {
                            "toolUseId": "tu_1",
                            "status": "success",
                            "content": [{"text": '{"status": "shipped"}'}],
                        }
                    },
                    {
                        "toolResult": {
                            "toolUseId": "tu_2",
                            "status": "success",
                            "content": [{"text": '{"status": "delayed"}'}],
                        }
                    },
                ],
            },
            {"role": "assistant", "content": [{"text": "Results ready."}]},
        ]
        record = normalizer.normalize(messages, "case_003", "Check two orders")
        assert len(record.tool_calls) == 2
        assert record.tool_calls[0].call_index == 0
        assert record.tool_calls[1].call_index == 1
        assert record.tool_calls[0].output == {"status": "shipped"}
        assert record.tool_calls[1].output == {"status": "delayed"}


class TestTraceResilience:
    def test_tool_result_in_non_user_message(self):
        """toolResult in a non-user role should still be captured."""
        messages = [
            {"role": "user", "content": [{"text": "Check order"}]},
            {
                "role": "assistant",
                "content": [
                    {
                        "toolUse": {
                            "toolUseId": "tu_1",
                            "name": "lookup_order_status",
                            "input": {"order_id": "ORD-1001"},
                        }
                    }
                ],
            },
            {
                "role": "tool",
                "content": [
                    {
                        "toolResult": {
                            "toolUseId": "tu_1",
                            "status": "success",
                            "content": [{"text": '{"status": "shipped"}'}],
                        }
                    }
                ],
            },
            {"role": "assistant", "content": [{"text": "Shipped."}]},
        ]
        record = normalizer.normalize(messages, "case_004", "Check order")
        assert record.tool_calls[0].output == {"status": "shipped"}

    def test_string_content(self):
        """Plain string content should not crash and should be captured."""
        messages = [
            {"role": "user", "content": "Hello there"},
            {"role": "assistant", "content": [{"text": "Hi!"}]},
        ]
        record = normalizer.normalize(messages, "case_005", "Hello there")
        assert len(record.conversation_turns) == 2
        assert record.conversation_turns[0].content == "Hello there"
        assert record.final_output == "Hi!"

    def test_missing_tool_result(self):
        """A toolUse with no matching toolResult should have None output."""
        messages = [
            {"role": "user", "content": [{"text": "Check order"}]},
            {
                "role": "assistant",
                "content": [
                    {
                        "toolUse": {
                            "toolUseId": "tu_orphan",
                            "name": "lookup_order_status",
                            "input": {"order_id": "ORD-1001"},
                        }
                    }
                ],
            },
            {"role": "assistant", "content": [{"text": "Something went wrong."}]},
        ]
        record = normalizer.normalize(messages, "case_006", "Check order")
        assert record.tool_calls[0].output is None


class TestExtractToolOutput:
    def test_json_block(self):
        result = normalizer._extract_tool_output(
            [{"json": {"status": "shipped"}}]
        )
        assert result == {"status": "shipped"}

    def test_text_block_json_parseable(self):
        result = normalizer._extract_tool_output(
            [{"text": '{"status": "shipped"}'}]
        )
        assert result == {"status": "shipped"}

    def test_text_block_plain(self):
        result = normalizer._extract_tool_output(
            [{"text": "some plain text"}]
        )
        assert result == "some plain text"

    def test_none_input(self):
        result = normalizer._extract_tool_output(None)
        assert result is None


class TestFinalOutput:
    def test_last_assistant_text(self):
        messages = [
            {"role": "user", "content": [{"text": "Hi"}]},
            {"role": "assistant", "content": [{"text": "First response."}]},
            {"role": "user", "content": [{"text": "More?"}]},
            {"role": "assistant", "content": [{"text": "Second response."}]},
        ]
        record = normalizer.normalize(messages, "case_007", "Hi")
        assert record.final_output == "Second response."


class TestPydanticDefaults:
    def test_mutable_defaults_are_independent(self):
        """Pydantic v2 deep-copies mutable defaults per instance."""
        from schemas.case import CaseExpectation

        a = CaseExpectation()
        b = CaseExpectation()
        a.allowed_tools.append("some_tool")
        assert b.allowed_tools == []
