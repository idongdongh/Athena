import unittest

from users import unique_user_ids


class UniqueUserIdsTests(unittest.TestCase):
    def test_first_seen_order_is_preserved(self):
        self.assertEqual([3, 1, 2], unique_user_ids([3, 1, 3, 2, 1]))

    def test_empty_input_is_empty(self):
        self.assertEqual([], unique_user_ids([]))


if __name__ == "__main__":
    unittest.main()
