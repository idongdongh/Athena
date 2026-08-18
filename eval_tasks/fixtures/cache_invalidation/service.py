from repository import UserRepository


class UserService:
    def __init__(self, repository: UserRepository):
        self.repository = repository
        self._cache: dict[int, str] = {}

    def get_name(self, user_id: int) -> str:
        if user_id not in self._cache:
            self._cache[user_id] = self.repository.get_name(user_id)
        return self._cache[user_id]

    def update_name(self, user_id: int, name: str) -> None:
        self.repository.update_name(user_id, name)
