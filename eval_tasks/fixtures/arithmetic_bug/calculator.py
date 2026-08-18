def percentage(value: float, total: float) -> float:
    """Return value as a percentage of total."""
    if total == 0:
        raise ValueError("total must not be zero")
    return total / value * 100
