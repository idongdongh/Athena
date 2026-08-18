import unittest

from retry import should_retry


class RetryBudgetTests(unittest.TestCase):
    def test_retry_is_allowed_before_budget_is_exhausted(self):
        self.assertTrue(should_retry(2, 3))

    def test_retry_is_denied_at_budget(self):
        self.assertFalse(should_retry(3, 3))

    def test_invalid_counters_are_rejected(self):
        with self.assertRaises(ValueError):
            should_retry(-1, 3)


if __name__ == "__main__":
    unittest.main()
