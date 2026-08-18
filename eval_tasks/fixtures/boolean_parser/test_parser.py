import unittest

from parser import parse_bool


class ParseBoolTests(unittest.TestCase):
    def test_true_values(self):
        for value in ("true", "TRUE", "1", "yes", "on"):
            self.assertTrue(parse_bool(value))

    def test_false_values(self):
        for value in ("false", "FALSE", "0", "no", "off"):
            self.assertFalse(parse_bool(value))

    def test_whitespace_is_ignored(self):
        self.assertFalse(parse_bool("  false  "))

    def test_unknown_value_is_rejected(self):
        with self.assertRaises(ValueError):
            parse_bool("sometimes")


if __name__ == "__main__":
    unittest.main()
