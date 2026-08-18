def page_items(items: list, page: int, page_size: int) -> list:
    if page < 1:
        raise ValueError("page must be at least 1")
    if page_size < 1:
        raise ValueError("page_size must be positive")
    start = page * page_size
    return items[start:start + page_size]
