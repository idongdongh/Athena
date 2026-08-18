import unittest

from repository import UserRepository
from service import UserService


class UserServiceTests(unittest.TestCase):
    def test_update_is_visible_after_value_was_cached(self):
        service = UserService(UserRepository())
        self.assertEqual("Ada", service.get_name(1))
        service.update_name(1, "Grace")
        self.assertEqual("Grace", service.get_name(1))

    def test_initial_read_uses_repository(self):
        self.assertEqual("Ada", UserService(UserRepository()).get_name(1))


if __name__ == "__main__":
    unittest.main()
