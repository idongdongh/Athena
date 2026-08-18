import unittest

from models import CartItem
from totals import subtotal


class SubtotalTests(unittest.TestCase):
    def test_quantity_contributes_to_subtotal(self):
        items = [CartItem(12.5, 2), CartItem(5.0, 3)]
        self.assertEqual(40.0, subtotal(items))

    def test_empty_cart_is_zero(self):
        self.assertEqual(0.0, subtotal([]))


if __name__ == "__main__":
    unittest.main()
