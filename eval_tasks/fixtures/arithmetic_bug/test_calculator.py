import unittest

from calculator import percentage


class PercentageTests(unittest.TestCase):
    def test_percentage_uses_value_over_total(self):
        self.assertEqual(25.0, percentage(50, 200))

    def test_zero_total_is_rejected(self):
        with self.assertRaises(ValueError):
            percentage(1, 0)


if __name__ == "__main__":
    unittest.main()
