"""Rouge Testing
"""

import unittest
from .rouge import RougeN


class TestRougeN(unittest.TestCase):
    def test_rouge_1(self):
        test_refs = [
            [
                "Claire walks her dog",
                "We had a three-course meal"
            ],
            [
                "Brad came to dinner with us",
                "He loves fish tacos"
            ]
        ]
        rouge_1 = RougeN(n_gram=1, refs=test_refs)
        self.assertEqual(
            rouge_1.score("Claire had a hot dog as her meal"),
            2/3
        )

    def test_rouge_2(self):
        test_refs = [
            [
                "Claire walks her dog",
                "We had a three-course meal"
            ],
            [
                "Brad came to dinner with us",
                "He loves fish tacos"
            ]
        ]
        rouge_1 = RougeN(n_gram=2, refs=test_refs)
        self.assertEqual(
            rouge_1.score("Claire had a hot dog as her meal"),
            0.4375
        )
