import unittest
from typing import Any

import pandas as pd

from pargraph import GraphEngine, delayed, graph


class TestGraphGeneration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.engine = GraphEngine()

    def test_basic(self):
        @delayed
        def add(x: int, y: int) -> int:
            return x + y

        @graph
        def sample_graph(w: int, x: int, y: int, z: int) -> int:
            return add(add(w, x), add(y, z))

        self.assertEqual(
            self.engine.get(*sample_graph.to_graph().to_dask(w=1, x=2, y=3, z=4))[0], sample_graph(w=1, x=2, y=3, z=4)
        )

    def test_basic_partial(self):
        @delayed
        def add(x: int, y: int) -> int:
            return x + y

        @graph
        def sample_graph(w: int, x: int, y: int, z: int) -> int:
            return add(add(w, x), add(y, z))

        self.assertEqual(
            self.engine.get(*sample_graph.to_graph(w=1, x=2).to_dask(y=3, z=4))[0], sample_graph(w=1, x=2, y=3, z=4)
        )

    def test_operator_override(self):
        @graph
        def sample_graph(w: int, x: int, y: int, z: int) -> int:
            return (w + x) + (y + z)

        self.assertEqual(
            self.engine.get(*sample_graph.to_graph().to_dask(w=1, x=2, y=3, z=4))[0], sample_graph(w=1, x=2, y=3, z=4)
        )

    def test_operator_override_complex(self):
        @graph
        def fibonacci(n: int) -> int:
            phi = (1 + 5**0.5) / 2
            return round(((phi**n) + ((1 - phi) ** n)) / 5**0.5)

        self.assertEqual(self.engine.get(*fibonacci.to_graph().to_dask(n=6))[0], fibonacci(n=6))

    def test_getitem(self):
        @delayed
        def return_tuple(x: int, y: int) -> Any:
            return x, y

        @graph
        def sample_graph(x: int, y: int) -> int:
            return return_tuple(x, y)[0]

        self.assertEqual(self.engine.get(*sample_graph.to_graph().to_dask(x=1, y=2))[0], sample_graph(x=1, y=2))

    def test_call(self):
        @graph
        def sample_graph(s: pd.Series) -> int:
            return s.sum()

        self.assertEqual(
            self.engine.get(*sample_graph.to_graph().to_dask(s=pd.Series([1, 2, 3])))[0],
            sample_graph(s=pd.Series([1, 2, 3])),
        )

    def test_call_complex(self):
        @graph
        def sample_graph(s: pd.Series) -> pd.Series:
            return s[s > s.mean()]

        pd.testing.assert_series_equal(
            self.engine.get(*sample_graph.to_graph().to_dask(s=pd.Series([1, 2, 3])))[0],
            sample_graph(s=pd.Series([1, 2, 3])),
        )

    def test_call_kw(self):
        @graph
        def sample_graph(df1: pd.DataFrame, df2: pd.DataFrame) -> pd.DataFrame:
            return df1.merge(df2, how="inner", on="a")

        pd.testing.assert_frame_equal(
            self.engine.get(
                *sample_graph.to_graph().to_dask(
                    df1=pd.DataFrame({"a": ["foo", "bar"], "b": [1, 2]}),
                    df2=pd.DataFrame({"a": ["foo", "baz"], "c": [3, 4]}),
                )
            )[0],
            sample_graph(
                df1=pd.DataFrame({"a": ["foo", "bar"], "b": [1, 2]}),
                df2=pd.DataFrame({"a": ["foo", "baz"], "c": [3, 4]}),
            ),
        )
