def unique_user_ids(user_ids: list[int]) -> list[int]:
    """Remove duplicate IDs without changing first-seen order."""
    return sorted(set(user_ids))
