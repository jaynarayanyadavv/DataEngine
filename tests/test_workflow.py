import unittest
from unittest.mock import patch

import pandas as pd
from pandas.testing import assert_frame_equal

from dataengine import clean, profile


def review(frame, *answers):
    replies = iter(answers)
    with patch("builtins.print"):
        return clean(frame, ask=lambda _: next(replies))


class Workflow(unittest.TestCase):
    def test_approval_and_original(self):
        frame = pd.DataFrame({"amount": ["1", "1", " ", "3"], "zip": ["001"] * 4})
        original = frame.copy()
        result = review(frame, "y")
        self.assertEqual(result.amount.tolist(), [1., 1., 3.])
        self.assertEqual(result.zip.tolist(), ["001"] * 3)
        assert_frame_equal(frame, original)

    def test_cancel_and_customize(self):
        frame = pd.DataFrame({"a": [1, 1, None], "b": [True] * 3})
        assert_frame_equal(review(frame, "n"), frame)
        assert_frame_equal(review(frame, "c", "2", "n"), frame)
        result = review(frame, "c", "99", "c", "1", "y")
        assert_frame_equal(result, frame.iloc[[0, 2]])

    def test_dependencies(self):
        frame = pd.DataFrame({"v": ["1", " ", "3"]})
        result = review(frame, "c", "3", "c", "1,2,3", "y")
        self.assertEqual(result.v.tolist(), [1., 2., 3.])

    def test_schema_edges(self):
        for frame in (pd.DataFrame(), pd.DataFrame(columns=["x"]), pd.DataFrame({
            "nested": [[1], {"a": 2}], "when": pd.to_datetime([None, "2024-01-01"]),
            "unknown": [None, None]})):
            assert_frame_equal(review(frame), frame)
        frame = pd.DataFrame([[None, "a"], [2., "b"]], columns=["x", "x"])
        self.assertEqual(review(frame, "y").iloc[0, 0], 2.)

    def test_outliers_modes_and_nullable_integers(self):
        frame = pd.DataFrame({"x": pd.array([1,2,3,4,5,6,7,100,None], dtype="Int64"),
                              "kind": pd.Categorical(["a"]*8 + [None])})
        result = review(frame, "y")
        self.assertEqual(result.x.iloc[-1], 4.5)
        self.assertLess(result.x.max(), 100)
        self.assertEqual(result.kind.iloc[-1], "a")
        self.assertIn("operations proposed", str(profile(frame)))
