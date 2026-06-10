import unittest
import sympy as sp
import numpy as np
from otaf.common import get_symbol_coef_map, get_symbols_in_expressions

class TestLatestFunctions(unittest.TestCase):
    def test_get_symbol_coef_map(self):
        # Test case 1: Basic test with one variable
        expr1 = sp.sympify("3*x + 2")
        self.assertEqual(get_symbol_coef_map(expr1), {"x": 3.0, "CONST": 2.0})

        # Test case 2: Test with multiple variables
        expr2 = sp.sympify("2*x + 5*y - 7*z + 1")
        self.assertEqual(
            get_symbol_coef_map(expr2), {"x": 2.0, "y": 5.0, "z": -7.0, "CONST": 1.0}
        )

        # Test case 3: Test with rounding
        expr3 = sp.sympify("1/3 * x")
        self.assertEqual(get_symbol_coef_map(expr3, rnd=2), {"x": 0.33, "CONST": 0.0})

        # Test case 4: Test with no variables (constant only)
        expr4 = sp.sympify("5.7")
        self.assertEqual(get_symbol_coef_map(expr4), {"CONST": 5.7})

    def test_get_symbols_in_expressions(self):
        expr_list1 = [sp.sympify("u_d_1 + v_d_2"), sp.sympify("w_g_1")]
        deviation_symbols, gap_symbols = get_symbols_in_expressions(expr_list1)
        self.assertEqual(deviation_symbols, [sp.Symbol("u_d_1"), sp.Symbol("v_d_2")])
        self.assertEqual(gap_symbols, [sp.Symbol("w_g_1")])


if __name__ == "__main__":
    unittest.main()
