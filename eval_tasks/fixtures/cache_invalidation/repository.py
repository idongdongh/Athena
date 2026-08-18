class UserRepository:
    def __init__(self):
        self._names = {1: "Ada"}

    def get_name(self, user_id: int) -> str:
        return self._names[user_id]

    def update_name(self, user_id: int, name: str) -> None:
        self._names[user_id] = name
