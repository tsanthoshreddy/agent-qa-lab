ORDER_FIXTURES: dict[str, dict] = {
    "ORD-1001": {"status": "shipped", "eta": "2026-03-22", "item": "Wireless Headphones"},
    "ORD-1002": {"status": "delayed", "eta": None, "item": "USB-C Hub"},
    "ORD-1003": {"status": "processing", "eta": "2026-03-25", "item": "Laptop Stand"},
    "ORD-1004": {"status": "delivered", "eta": None, "delivered_date": "2026-03-15", "item": "Keyboard"},
    "ORD-1005": {"status": "cancelled", "eta": None, "item": "Mouse Pad"},
    "ORD-1006": {"status": "shipped", "eta": "2026-03-21", "item": "Monitor Arm"},
}

CUSTOMER_FIXTURES: dict[str, dict] = {
    "CUST-100": {
        "name": "Alice",
        "email": "alice@example.com",
        "order_ids": ["ORD-1001", "ORD-1003"],
    },
    "CUST-200": {
        "name": "Bob",
        "email": "bob@example.com",
        "order_ids": ["ORD-1002"],
    },
}
