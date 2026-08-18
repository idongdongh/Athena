import unittest

from pagination import page_items


class PaginationTests(unittest.TestCase):
    def test_first_page_starts_at_first_item(self):
        self.assertEqual([1, 2], page_items([1, 2, 3, 4], 1, 2))

    def test_second_page_follows_first_page(self):
        self.assertEqual([3, 4], page_items([1, 2, 3, 4], 2, 2))

    def test_invalid_page_is_rejected(self):
        with self.assertRaises(ValueError):
            page_items([1], 0, 1)


if __name__ == "__main__":
    unittest.main()
