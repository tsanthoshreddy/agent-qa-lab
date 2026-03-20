import os

from strands import Agent
from strands.models import OpenAIModel

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


def create_support_agent() -> Agent:
    model_id = os.environ.get("AQL_MODEL_ID", "gpt-4o")
    model = OpenAIModel(
        client_args={"api_key": os.environ["OPENAI_API_KEY"]},
        model_id=model_id,
        params={"temperature": 0},
    )
    return Agent(
        model=model,
        tools=TOOLS,
        system_prompt=SYSTEM_PROMPT,
        callback_handler=None,
    )
