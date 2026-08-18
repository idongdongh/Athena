def should_retry(attempts_completed: int, max_attempts: int) -> bool:
    if attempts_completed < 0 or max_attempts < 1:
        raise ValueError("invalid retry counters")
    return attempts_completed <= max_attempts
