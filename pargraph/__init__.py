from .about import __version__
from .engine.engine import GraphEngine
from .graph.annotation import Result
from .graph.decorators import delayed, graph

assert isinstance(__version__, str)
assert isinstance(GraphEngine, type)
assert isinstance(Result, type)
assert callable(delayed)
assert callable(graph)
