from pydantic import BaseModel, ConfigDict


class CaseExpectation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    allowed_tools: list[str] = []
    required_tools: list[str] = []
    forbidden_tools: list[str] = []
    required_arguments: dict[str, dict[str, str]] = {}
    expected_constraints: list[str] = []


class TestCase(BaseModel):
    model_config = ConfigDict(extra="forbid")

    case_id: str
    input: str
    category: str
    expectation: CaseExpectation
