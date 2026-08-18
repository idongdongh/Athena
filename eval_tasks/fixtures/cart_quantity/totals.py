from models import CartItem


def subtotal(items: list[CartItem]) -> float:
    return sum(item.unit_price for item in items)
