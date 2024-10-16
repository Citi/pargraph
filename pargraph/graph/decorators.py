import dataclasses
import functools
import inspect
import itertools
import operator
import uuid
import warnings
from dataclasses import dataclass
from typing import Any, Callable, Optional, Protocol, Tuple, Union, cast, Iterator

from pargraph.graph.annotation import _get_output_names
from pargraph.graph.objects import (
    Const,
    ConstKey,
    FunctionCall,
    Graph,
    GraphCall,
    InputKey,
    NodeKey,
    NodeOutputKey,
    OutputKey,
)


class Graphable(Protocol):
    def __call__(self, *args, **kwargs) -> Any:
        """
        Call the function with the given arguments

        :param args: positional arguments to the function
        :param kwargs: keyword arguments to the function
        :return: result of the function
        """
        ...

    def to_graph(self, *args, **kwargs) -> Graph:
        """
        Generate a graph from the function and its materialized arguments

        :param args: positional arguments to the function
        :param kwargs: keyword arguments to the function
        :return: generated graph
        """
        ...


@dataclass
class GraphContext:
    """
    GraphContext holds an intermediary graph and a target key

    If target is None, the GraphContext represents an external input
    """

    _graph: Graph
    _target: Optional[Union[ConstKey, InputKey, NodeOutputKey]]

    """
    Common magic methods
    """

    def __getattr__(self, item):
        if item in {field.name for field in dataclasses.fields(self)}:
            return getattr(self, item)

        def _getattr(self, item) -> Any:
            return getattr(self, item)

        _getattr.__name__ = "getattr"
        return _implicit_delayed(_getattr)(self, item)

    def __call__(self, *args, **kwargs):
        warnings.warn(
            "Calling a GraphContext object is not recommended, please use a dedicated delayed function instead",
            stacklevel=2,
        )

        @delayed
        def call(*args) -> Any:
            func, arg_length, *args = args  # type: ignore[assignment]
            positional_args = args[:arg_length]
            keyword_args = args[arg_length:]
            return func(*positional_args, **dict(zip(keyword_args[0::2], keyword_args[1::2])))

        return call(self, len(args), *args, *itertools.chain(*kwargs.items()))


def external_input() -> GraphContext:
    """
    Create an external input

    :return: GraphContext with an empty graph and no target
    """
    return GraphContext(_graph=Graph(consts={}, inputs={}, nodes={}, outputs={}), _target=None)


def graph(function: Callable) -> Graphable:
    """
    Graph decorator

    .. note::

        There are a few assumptions when decorating a function with the graph decorator:

        - The function must have a return annotation
        - Variadic positional and/or keyword arguments are not supported
        - Function must not have any side effects
        - Function must be comprised of other graph or delayed functions

    :param function: function to decorate
    :return: decorated function
    """
    signature = inspect.signature(function)
    for param in signature.parameters.values():
        if param.kind in {inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD}:
            raise ValueError("Variadic positional and/or keyword arguments are not supported")

    @functools.wraps(function)
    def wrapper(*args, **kwargs) -> Union[Graph, GraphContext, Tuple[GraphContext, ...], Any]:
        # Bind arguments to handle for positional arguments
        bound_args = inspect.signature(function).bind(*args, **kwargs)

        # If all arguments are not GraphContext, compute function directly
        if all(not isinstance(arg, GraphContext) for arg in bound_args.arguments.values()):
            return function(*args, **kwargs)

        # Generate graph context
        # NOTE: Pass a graph context with only one input for each arg so that it is merged into the graph downstream
        graph_result = function(
            **{
                name: (
                    GraphContext(
                        _graph=Graph(consts={}, inputs={InputKey(key=name): None}, nodes={}, outputs={}),
                        _target=InputKey(key=name),
                    )
                    if isinstance(arg, GraphContext)
                    else arg
                )
                for name, arg in bound_args.arguments.items()
            }
        )

        output_names = _get_output_names(function)

        # Extract sub graphs from the generated graph context and merge it into one sub graph
        sub_graph = _merge_graphs(
            *(
                Graph(
                    consts=graph_context._graph.consts,
                    inputs=graph_context._graph.inputs,
                    nodes=graph_context._graph.nodes,
                    outputs={OutputKey(key=output_name): graph_context._target},
                )
                for output_name, graph_context in cast(
                    Iterator[Tuple[str, GraphContext]],
                    (
                        zip(output_names, graph_result)
                        if isinstance(output_names, tuple)
                        else ((output_names, graph_result),)
                    ),
                )
                if isinstance(graph_context, GraphContext)
            )
        )

        # Short circuit if external input is passed in (for top-level graph calls)
        if any(arg._target is None for arg in bound_args.arguments.values() if isinstance(arg, GraphContext)):
            # Inject default values for external inputs
            for name, arg in bound_args.arguments.items():
                default_value = bound_args.signature.parameters[name].default
                if default_value is inspect.Parameter.empty:
                    continue

                const_id = f"_{uuid.uuid4().hex}"
                sub_graph.consts[ConstKey(key=const_id)] = Const.from_value(default_value)
                sub_graph.inputs[InputKey(key=name)] = ConstKey(key=const_id)

            return sub_graph

        # Inject sub graph into parent graph
        node_id = f"{function.__name__}_{uuid.uuid4().hex}"
        parent_graph = _merge_graphs(
            *(arg._graph for arg in bound_args.arguments.values() if isinstance(arg, GraphContext))
        )
        parent_graph.nodes[NodeKey(key=node_id)] = GraphCall(
            graph=sub_graph,
            graph_name=function.__name__,
            args={name: arg._target for name, arg in bound_args.arguments.items() if isinstance(arg, GraphContext)},
        )

        return (
            tuple(
                GraphContext(_graph=parent_graph, _target=NodeOutputKey(key=node_id, output=output_name))
                for output_name in output_names
            )
            if isinstance(output_names, tuple)
            else GraphContext(_graph=parent_graph, _target=NodeOutputKey(key=node_id, output=output_names))
        )

    def to_graph(*args, **kwargs):
        return _generate_graph(wrapper, *args, **kwargs)

    wrapper.to_graph = to_graph  # type: ignore[attr-defined]

    return cast(Graphable, wrapper)


def delayed(function: Callable) -> Graphable:
    """
    Delayed decorator

    .. note::

        There are a few assumptions when decorating a function with the delayed decorator:

        - The function must have a return annotation
        - Variadic positional arguments are supported, but must not contain other named arguments
        - Function must not have any side effects

    :param function: function to decorate
    :return: decorated function
    """
    signature = inspect.signature(function)
    for param in signature.parameters.values():
        if param.kind == inspect.Parameter.VAR_KEYWORD:
            raise ValueError("Variadic keyword arguments are not supported")

        if param.kind == inspect.Parameter.VAR_POSITIONAL and len(signature.parameters) > 1:
            raise ValueError("Variadic positional arguments are only supported if no other named arguments are present")

    @functools.wraps(function)
    def wrapper(*args, **kwargs) -> Union[Graph, GraphContext, Tuple[GraphContext, ...], Any]:
        # If all arguments are not GraphContext, compute function directly
        # Note: cannot use bound_args because arg may be variadic
        if all(not isinstance(arg, GraphContext) for arg in itertools.chain(args, kwargs.values())):
            return function(*args, **kwargs)

        # Bind arguments to handle for positional arguments
        bound_args = inspect.signature(function).bind(*args, **kwargs)

        # Handle variadic positional arguments
        if next(iter(bound_args.signature.parameters.values())).kind == inspect.Parameter.VAR_POSITIONAL:
            arg_dict = {
                str(i): arg if isinstance(arg, GraphContext) else _create_const(arg)
                for i, arg in enumerate(bound_args.args)
            }
        # Handle regular positional and keyword arguments
        else:
            arg_dict = {
                name: arg if isinstance(arg, GraphContext) else _create_const(arg)
                for name, arg in bound_args.arguments.items()
            }

        output_names = _get_output_names(function)

        # Short circuit if external input is passed in (for top-level delayed calls)
        if any(graph_context._target is None for graph_context in arg_dict.values()):
            graph_result = Graph(
                consts={
                    ConstKey(key=name): arg._graph.consts[arg._target]
                    for name, arg in arg_dict.items()
                    if isinstance(arg._target, ConstKey)
                },
                inputs={InputKey(key=name): None for name, arg in arg_dict.items() if arg._target is None},
                nodes={
                    NodeKey(key=function.__name__): FunctionCall(
                        function=function,
                        args={
                            name: InputKey(key=name) if arg._target is None else ConstKey(key=name)
                            for name, arg in arg_dict.items()
                        },
                    )
                },
                outputs={
                    OutputKey(key=output_name): NodeOutputKey(key=function.__name__, output=output_name)
                    for output_name in (output_names if isinstance(output_names, tuple) else (output_names,))
                },
            )

            # Inject default values for external inputs
            for name, arg in bound_args.arguments.items():
                default_value = bound_args.signature.parameters[name].default
                if default_value is inspect.Parameter.empty:
                    continue

                const_id = f"_{uuid.uuid4().hex}"
                graph_result.consts[ConstKey(key=const_id)] = Const.from_value(default_value)
                graph_result.inputs[InputKey(key=name)] = ConstKey(key=const_id)

            return graph_result

        # Inject function call node into graph
        node_id = f"{function.__name__}_{uuid.uuid4().hex}"
        merged_graph = _merge_graphs(*(arg._graph for arg in arg_dict.values()))
        merged_graph.nodes[NodeKey(key=node_id)] = FunctionCall(
            function=function, args={name: graph_context._target for name, graph_context in arg_dict.items()}
        )

        return (
            tuple(
                GraphContext(_graph=merged_graph, _target=NodeOutputKey(key=node_id, output=output_name))
                for output_name in output_names
            )
            if isinstance(output_names, tuple)
            else GraphContext(_graph=merged_graph, _target=NodeOutputKey(key=node_id, output=output_names))
        )

    def to_graph(*args, **kwargs):
        return _generate_graph(wrapper, *args, **kwargs)

    wrapper.to_graph = to_graph  # type: ignore[attr-defined]

    return cast(Graphable, wrapper)


def _merge_graphs(*graphs: Graph) -> Graph:
    graph = Graph(consts={}, inputs={}, nodes={}, outputs={})

    for g in graphs:
        graph.consts.update(g.consts)
        graph.inputs.update(g.inputs)
        graph.nodes.update(g.nodes)
        graph.outputs.update(g.outputs)

    return graph


def _create_const(value: Any) -> GraphContext:
    key = f"_{uuid.uuid4().hex}"
    return GraphContext(
        _graph=Graph(consts={ConstKey(key=key): Const.from_value(value)}, inputs={}, nodes={}, outputs={}),
        _target=ConstKey(key=key),
    )


def _implicit_delayed(function: Callable) -> Graphable:
    function.__implicit = True  # type: ignore[attr-defined]
    return delayed(function)


def _generate_graph(func: Callable, /, *args, **kwargs):
    """
    Generate a graph from a function and its materialized arguments.

    :param func: function to generate a graph from
    :param args: positional arguments to the function
    :param kwargs: keyword arguments to the function
    :return: generated graph
    """
    bound_args = inspect.signature(func).bind_partial(*args, **kwargs)

    new_args = []
    new_kwargs = {}
    for name, param in bound_args.signature.parameters.items():
        if param.kind == param.POSITIONAL_ONLY:
            new_args.append(bound_args.arguments.get(name, external_input()))
        elif param.kind in {param.POSITIONAL_OR_KEYWORD, param.KEYWORD_ONLY}:
            new_kwargs[name] = bound_args.arguments.get(name, external_input())
        else:
            raise ValueError(f"unsupported parameter kind: {param.kind}")

    return func(*new_args, **new_kwargs)


# Register unary operators, may be missing some
#
# __hash__() is not included because since Python 3.3 hash randomization is enabled by default yielding impure hashes
# between isolated instances making it not suitable for out-of-core computations
# More info: https://docs.python.org/3/reference/datamodel.html#object.__hash__
for op in {
    bool,
    int,
    float,
    complex,
    str,
    bytes,
    len,
    operator.neg,
    operator.pos,
    operator.abs,
    operator.invert,
    operator.index,
}:
    op = cast(Callable, op)

    def wrapper(op: Callable):
        def meth(self) -> Any:
            return op(self)

        # Set the name of the method to the name of the operator
        meth.__name__ = op.__name__

        return _implicit_delayed(meth)

    setattr(GraphContext, f"__{op.__name__.strip('_')}__", wrapper(op))

# Register unary operators that allow an optional parameter
for op in {round}:

    def wrapper(op: Callable):
        def meth(self, param=None) -> Any:
            return op(self, param)

        # Set the name of the method to the name of the operator
        meth.__name__ = op.__name__

        return _implicit_delayed(meth)

    setattr(GraphContext, f"__{op.__name__.strip('_')}__", wrapper(op))

# Register binary operators, may be missing some
for op in {
    operator.add,
    operator.sub,
    operator.mul,
    operator.matmul,
    operator.truediv,
    operator.floordiv,
    operator.mod,
    divmod,
    operator.lshift,
    operator.rshift,
    operator.and_,
    operator.xor,
    operator.or_,
    operator.lt,
    operator.le,
    operator.eq,
    operator.ne,
    operator.gt,
    operator.ge,
    operator.getitem,
    operator.contains,
    format,
}:
    op = cast(Callable, op)

    def wrapper(op: Callable):
        def meth(self, other) -> Any:
            return op(self, other)

        # Set the name of the method to the name of the operator
        meth.__name__ = op.__name__

        return _implicit_delayed(meth)

    setattr(GraphContext, f"__{op.__name__.strip('_')}__", wrapper(op))

# Register binary operators that allow an optional parameter
for op in {pow}:
    op = cast(Callable, op)

    def wrapper(op: Callable):
        def meth(self, other, param=None) -> Any:
            return op(self, other, param)

        # Set the name of the method to the name of the operator
        meth.__name__ = op.__name__

        return _implicit_delayed(meth)

    setattr(GraphContext, f"__{op.__name__.strip('_')}__", wrapper(op))

# Register right-hand side binary operators, may be missing some
for op in {
    operator.add,
    operator.sub,
    operator.mul,
    operator.matmul,
    operator.truediv,
    operator.floordiv,
    operator.mod,
    divmod,
    operator.lshift,
    operator.rshift,
    operator.and_,
    operator.xor,
    operator.or_,
}:
    op = cast(Callable, op)

    def wrapper(op: Callable):
        def meth(self, other) -> Any:
            return op(other, self)

        # Set the name of the method to the name of the operator
        meth.__name__ = op.__name__

        return _implicit_delayed(meth)

    setattr(GraphContext, f"__r{op.__name__.strip('_')}__", wrapper(op))

# Register right-hand side binary operators that allow an optional parameter
for op in {pow}:
    op = cast(Callable, op)

    def wrapper(op: Callable):
        def meth(self, other, param=None) -> Any:
            return op(other, self, param)

        # Set the name of the method to the name of the operator
        meth.__name__ = op.__name__

        return _implicit_delayed(meth)

    setattr(GraphContext, f"__r{op.__name__.strip('_')}__", wrapper(op))
