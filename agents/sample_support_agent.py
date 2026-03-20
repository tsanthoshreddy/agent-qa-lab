import os
import sys

from strands import Agent

from domain.support_tools import lookup_customer_info, lookup_order_status, refund_order

SYSTEM_PROMPT = """You are a customer support agent for an online store.
Your job is to help customers check order status.

Rules:
- Use the lookup_order_status tool when a customer asks about an order.
- Only use tools when you have a valid identifier to look up.
- If the customer does not provide an order ID, ask for it.
- Never guess or fabricate order information.
- Do not process refunds unless the customer explicitly asks for one.
- Base your answer strictly on the tool's returned data."""

TOOLS = [lookup_order_status, lookup_customer_info, refund_order]

# Provider -> default model ID
DEFAULT_MODELS = {
    "openai": "gpt-4o",
    "bedrock": "us.anthropic.claude-sonnet-4-20250514-v1:0",
}


def _build_model(provider: str, model_id: str):
    """Build a Strands model instance for the given provider."""
    if provider == "openai":
        from strands.models.openai import OpenAIModel

        client_args = {"api_key": os.environ["OPENAI_API_KEY"]}
        base_url = os.environ.get("OPENAI_BASE_URL")
        if base_url:
            client_args["base_url"] = base_url
        return OpenAIModel(
            client_args=client_args,
            model_id=model_id,
            params={"temperature": 0},
        )

    if provider == "bedrock":
        from strands.models.bedrock import BedrockModel

        return BedrockModel(model_id=model_id, temperature=0)

    print(
        f"Unknown provider '{provider}'. Supported: openai, bedrock",
        file=sys.stderr,
    )
    sys.exit(1)


def create_support_agent() -> Agent:
    provider = os.environ.get("AQL_PROVIDER", "openai").lower()
    model_id = os.environ.get("AQL_MODEL_ID", DEFAULT_MODELS.get(provider, "gpt-4o"))
    model = _build_model(provider, model_id)
    return Agent(
        model=model,
        tools=TOOLS,
        system_prompt=SYSTEM_PROMPT,
        callback_handler=None,
    )
