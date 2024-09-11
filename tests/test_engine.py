import unittest
import uuid
from concurrent.futures import Future
from operator import add
from typing import Any, Callable, Dict, Optional, Tuple, cast

from loky import get_reusable_executor

from pargraph import GraphEngine
from pargraph.engine.engine import Backend


def generate_binary_tree_graph(binary_op: Callable = add, leaf_value: Any = 1, max_height: int = 3) -> Tuple[Dict, str]:
    def create_binary_tree_node(parent: str, height: int):
        left = str(uuid.uuid4())
        right = str(uuid.uuid4())
        graph[parent] = (binary_op, left, right)

        if height <= 1:
            graph[left] = leaf_value
            graph[right] = leaf_value
            return

        create_binary_tree_node(left, height - 1)
        create_binary_tree_node(right, height - 1)

    graph: dict = {}
    root = str(uuid.uuid4())
    create_binary_tree_node(root, max_height)
    return graph, root


def add_one(func: Callable, *args) -> int:
    return func(*args) + 1


class TestEngineBase(unittest.TestCase):
    engine: Optional[GraphEngine] = None

    @classmethod
    def setUpClass(cls):
        raise unittest.SkipTest("abstract test class")

    def test_example(self):
        graph = {"x": 1, "y": 2, "z": (add, "x", "y"), "w": (sum, ["x", "y", "z"])}
        self.assertEqual(self.engine.get(graph, "x"), 1)
        self.assertEqual(self.engine.get(graph, "z"), 3)
        self.assertEqual(self.engine.get(graph, "w"), 6)
        self.assertEqual(self.engine.get(graph, ["x", "y", "z"]), [1, 2, 3])
        self.assertEqual(self.engine.get(graph, [["x", "y"], ["z", "w"]]), [[1, 2], [3, 6]])

    def test_cycle(self):
        graph = {"a": "b", "b": "c", "c": "a"}
        with self.assertRaises(ValueError):
            self.engine.get(graph, "a")

    def test_nested(self):
        graph = {"x": 1, "y": 2, "z": (add, (add, "x", "y"), (add, "x", "y"))}
        self.assertEqual(self.engine.get(graph, "z"), 6)

    def test_binary_tree_small(self):
        max_height = 3
        graph, root = generate_binary_tree_graph(binary_op=add, leaf_value=1, max_height=max_height)
        self.assertEqual(self.engine.get(graph, root), 2**max_height)

    def test_binary_tree_large(self):
        max_height = 8
        graph, root = generate_binary_tree_graph(binary_op=add, leaf_value=1, max_height=max_height)
        self.assertEqual(self.engine.get(graph, root), 2**max_height)


class TestEngineDefaultBackend(TestEngineBase):
    @classmethod
    def setUpClass(cls):
        cls.engine = GraphEngine(get_reusable_executor(max_workers=4))


class TestEngineCustomBackend(TestEngineBase):
    @classmethod
    def setUpClass(cls):
        class CustomBackend:
            def __init__(self):
                pass

            def submit(self, fn, /, *args, **kwargs) -> Future:
                future: Future[Any] = Future()
                future.set_result(fn(*args, **kwargs))
                return future

        cls.engine = GraphEngine(cast(Backend, CustomBackend()))


if __name__ == "__main__":
    unittest.main()
