from datetime import date


def overlaps(start_a: date, end_a: date, start_b: date, end_b: date) -> bool:
    if start_a > end_a or start_b > end_b:
        raise ValueError("range start must not follow range end")
    return start_a < end_b and start_b < end_a
