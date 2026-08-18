from dataclasses import dataclass


@dataclass(frozen=True)
class CartItem:
    unit_price: float
    quantity: int
