import unittest
from datetime import date

from ranges import overlaps


class RangeOverlapTests(unittest.TestCase):
    def test_touching_inclusive_ranges_overlap(self):
        self.assertTrue(
            overlaps(date(2026, 1, 1), date(2026, 1, 3), date(2026, 1, 3), date(2026, 1, 5))
        )

    def test_separated_ranges_do_not_overlap(self):
        self.assertFalse(
            overlaps(date(2026, 1, 1), date(2026, 1, 2), date(2026, 1, 3), date(2026, 1, 5))
        )

    def test_invalid_range_is_rejected(self):
        with self.assertRaises(ValueError):
            overlaps(date(2026, 1, 2), date(2026, 1, 1), date(2026, 1, 3), date(2026, 1, 5))


if __name__ == "__main__":
    unittest.main()
