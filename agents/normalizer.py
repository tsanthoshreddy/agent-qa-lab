import uuid
from typing import Any

from schemas.run_record import ConversationTurn, RunRecord, ToolCall


class StrandsTraceNormalizer:
    def __init__(self, agent_name: str = "support_agent_v1"):
        self.agent_name = agent_name

    def normalize(
        self, messages: list[dict[str, Any]], case_id: str, input_text: str
    ) -> RunRecord:
        conversation_turns: list[ConversationTurn] = []
        tool_calls: list[ToolCall] = []
        tool_call_index = 0

        # First pass: collect tool results keyed by toolUseId (from any role)
        tool_results_by_id: dict[str, Any] = {}
        for msg in messages:
            content = msg.get("content", [])
            if not isinstance(content, list):
                continue
            for block in content:
                if isinstance(block, dict) and "toolResult" in block:
                    tr = block["toolResult"]
                    tool_results_by_id[tr["toolUseId"]] = tr.get("content")

        # Second pass: build conversation turns and tool calls
        for msg in messages:
            role = msg["role"]
            content = msg.get("content", [])
            if not isinstance(content, list):
                # Handle plain string content
                if isinstance(content, str) and content:
                    conversation_turns.append(
                        ConversationTurn(role=role, content=content)
                    )
                continue
            for block in content:
                if not isinstance(block, dict):
                    continue
                if "text" in block:
                    conversation_turns.append(
                        ConversationTurn(role=role, content=block["text"])
                    )
                elif "toolUse" in block:
                    tu = block["toolUse"]
                    output = tool_results_by_id.get(tu["toolUseId"])
                    # Extract the actual value from toolResult content blocks
                    normalized_output = self._extract_tool_output(output)
                    tool_calls.append(
                        ToolCall(
                            tool_name=tu["name"],
                            arguments=tu.get("input", {}),
                            output=normalized_output,
                            call_index=tool_call_index,
                        )
                    )
                    tool_call_index += 1

        # final_output = last assistant text turn
        final_output = ""
        for turn in reversed(conversation_turns):
            if turn.role == "assistant":
                final_output = turn.content
                break

        return RunRecord(
            run_id=f"run_{uuid.uuid4().hex[:8]}",
            case_id=case_id,
            agent_name=self.agent_name,
            input_text=input_text,
            conversation_turns=conversation_turns,
            tool_calls=tool_calls,
            final_output=final_output,
            metadata={"tool_call_count": len(tool_calls)},
        )

    @staticmethod
    def _extract_tool_output(content: Any) -> Any:
        """Extract usable value from Strands toolResult content blocks.

        Strands wraps tool results as [{"text": "..."}, ...] or [{"json": {...}}, ...].
        Pull the inner value out so evaluators get clean data.
        """
        if content is None:
            return None
        if isinstance(content, list):
            for block in content:
                if isinstance(block, dict):
                    if "json" in block:
                        return block["json"]
                    if "text" in block:
                        # Try to parse JSON strings into dicts
                        text = block["text"]
                        try:
                            import json

                            return json.loads(text)
                        except (json.JSONDecodeError, TypeError):
                            return text
            # Fall back to the raw list if nothing matched
            return content
        return content
