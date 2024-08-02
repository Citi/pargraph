import sys
import unittest
from typing import Tuple

from pargraph.graph.annotation import Result, _get_output_names

if sys.version_info < (3, 9):
    from typing_extensions import Annotated
else:
    from typing import Annotated


class TestAnnotation(unittest.TestCase):
    def test_non_tuple_non_annotated_return_value(self):
        def test_function() -> int:
            return 1

        self.assertEqual(_get_output_names(test_function), "result")

    def test_tuple_non_annotated_return_value(self):
        def test_function() -> Tuple[int, int]:
            return 1, 2

        self.assertEqual(_get_output_names(test_function), ("result_0", "result_1"))

    def test_non_tuple_annotated_return_value(self):
        def test_function() -> Annotated[int, Result("result_a")]:
            return 1

        self.assertEqual(_get_output_names(test_function), "result_a")

    def test_tuple_annotated_return_value(self):
        def test_function() -> Tuple[Annotated[int, Result("result_a")], Annotated[int, Result("result_b")]]:
            return 1, 2

        self.assertEqual(_get_output_names(test_function), ("result_a", "result_b"))

    def test_duplicate_result_name(self):
        def test_function() -> Tuple[Annotated[int, Result("result_a")], Annotated[int, Result("result_a")]]:
            return 1, 2

        with self.assertRaises(ValueError):
            _get_output_names(test_function)

    def test_multiple_results_invalid_annotation(self):
        def test_function() -> Annotated[int, Result("result_a"), Result("result_b")]:
            return 1

        with self.assertRaises(ValueError):
            _get_output_names(test_function)
