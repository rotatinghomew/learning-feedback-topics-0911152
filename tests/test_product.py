from collections import Counter
import json
from pathlib import Path
import unittest

from learning.domain import overview, validate


class ProductTests(unittest.TestCase):
    def setUp(self):
        self.tables = json.loads(Path("data/releases.json").read_text(encoding="utf-8"))["term-a"]

    def test_word_counts_preserve_actual_tokens(self):
        result = overview(self.tables, {})
        for summary in result["terms"]:
            expected = Counter(word for row in self.tables["feedback"] if row["term"] == summary["term"] for word in row["text"].split())
            self.assertEqual(dict(summary["words"]), dict(expected))

    def test_terminology_is_not_silently_applied_to_raw_text(self):
        before = overview(self.tables, {})
        self.tables["terminology"] = []
        self.assertEqual(before, overview(self.tables, {}))

    def test_empty_document_rejected(self):
        self.tables["feedback"][0]["text"] = " "
        with self.assertRaises(ValueError):
            validate(self.tables)
