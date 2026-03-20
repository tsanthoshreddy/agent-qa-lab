from typing import Any

from pydantic import BaseModel


class ConversationTurn(BaseModel):
    role: str
    content: str
    timestamp: str | None = None


class ToolCall(BaseModel):
    tool_name: str
    arguments: dict[str, Any]
    output: Any | None = None
    timestamp: str | None = None
    call_index: int


class RunRecord(BaseModel):
    run_id: str
    case_id: str
    agent_name: str
    input_text: str
    conversation_turns: list[ConversationTurn]
    tool_calls: list[ToolCall]
    final_output: str
    raw_trace_path: str | None = None
    metadata: dict[str, Any] = {}
