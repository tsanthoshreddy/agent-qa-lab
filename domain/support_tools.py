from strands import tool

from domain.fixtures import CUSTOMER_FIXTURES, ORDER_FIXTURES


@tool
def lookup_order_status(order_id: str) -> dict:
    """Look up the current status of a customer order.

    Returns shipping status, estimated delivery date, and item details
    for the given order ID.
    """
    if order_id not in ORDER_FIXTURES:
        return {"error": f"Order {order_id} not found"}
    return ORDER_FIXTURES[order_id]


@tool
def lookup_customer_info(customer_id: str) -> dict:
    """Look up customer account information.

    Returns the customer's name, email, and associated order IDs.
    """
    if customer_id not in CUSTOMER_FIXTURES:
        return {"error": f"Customer {customer_id} not found"}
    return CUSTOMER_FIXTURES[customer_id]


@tool
def refund_order(order_id: str, reason: str) -> dict:
    """Process a refund for the given order.

    This initiates a refund and returns confirmation. Only use when
    a customer explicitly requests a refund.
    """
    return {"status": "refund_initiated", "order_id": order_id, "reason": reason}
