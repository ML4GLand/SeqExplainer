import unittest

def add_two_numbers(a, b):
    return a + b

class TestAddTwoNumbers(unittest.TestCase):
    def test_add_two_numbers_positive(self):
        self.assertEqual(add_two_numbers(2, 3), 5)

    def test_add_two_numbers_negative(self):
        self.assertEqual(add_two_numbers(-2, -3), -5)

    def test_add_two_numbers_zero(self):
        self.assertEqual(add_two_numbers(0,