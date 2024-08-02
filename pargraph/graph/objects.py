import base64
import copy
import inspect
import re
import uuid
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Union

import cloudpickle
import jsonschema
import msgpack
import pydot

from pargraph.graph.annotation import _get_output_names

_key_pattern = r"^[a-zA-Z0-9_]+(\.[a-zA-Z0-9_]+)*$"

_graph_json_schema = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "title": "JSON Graph Schema",
    "type": "object",
    "properties": {
        "consts": {
            "type": "object",
            "patternProperties": {
                r"^[a-zA-Z0-9_]+(\.[a-zA-Z0-9_]+)*$": {
                    "type": "object",
                    "properties": {"type": {"type": "string"}, "value": {"type": "string"}},
                }
            },
        },
        "nodes": {
            "type": "object",
            "patternProperties": {
                r"^[a-zA-Z0-9_]+(\.[a-zA-Z0-9_]+)*$": {
                    "anyOf": [
                        {"type": "object", "properties": {"function": {"type": "string"}}},
                        {"type": "object", "properties": {"graph": {"$ref": "#"}}},
                    ]
                }
            },
        },
        "edges": {
            "type": "array",
            "items": {
                "type": "array",
                "prefixItems": [
                    {
                        "anyOf": [
                            {"type": "string", "pattern": r"^consts:[a-zA-Z0-9_]+(\.[a-zA-Z0-9_]+)*$"},
                            {"type": "string", "pattern": r"^inputs:[a-zA-Z0-9_]+(\.[a-zA-Z0-9_]+)*$"},
                            {
                                "type": "string",
                                "pattern": r"^nodes:[a-zA-Z0-9_]+(\.[a-zA-Z0-9_]+)*:outputs:"
                                r"[a-zA-Z0-9_]+(\.[a-zA-Z0-9_]+)*$",
                            },
                        ]
                    },
                    {
                        "type": "string",
                        "pattern": r"^nodes:[a-zA-Z0-9_]+(\.[a-zA-Z0-9_]+)*:inputs:[a-zA-Z0-9_]+(\.[a-zA-Z0-9_]+)*$",
                    },
                ],
            },
        },
        "outputs": {
            "type": "object",
            "patternProperties": {
                r"^[a-zA-Z0-9_]+(\.[a-zA-Z0-9_]+)*$": {
                    "anyOf": [
                        {"type": "string", "pattern": r"^consts:[a-zA-Z0-9_]+(\.[a-zA-Z0-9_]+)*$"},
                        {"type": "string", "pattern": r"^inputs:[a-zA-Z0-9_]+(\.[a-zA-Z0-9_]+)*$"},
                        {
                            "type": "string",
                            "pattern": r"^nodes:[a-zA-Z0-9_]+(\.[a-zA-Z0-9_]+)*:outputs:"
                            r"[a-zA-Z0-9_]+(\.[a-zA-Z0-9_]+)*$",
                        },
                    ]
                }
            },
        },
    },
}


@dataclass(frozen=True)
class Const:
    type: str
    value: str

    def __post_init__(self):
        assert isinstance(self.type, str), f"Type must be a string; got type '{type(self.type)}'"
        assert isinstance(self.value, str), f"Value must be a string; got type '{type(self.value)}'"

    @staticmethod
    def from_dict(data: Dict) -> "Const":
        return Const(**data)

    def to_dict(self) -> Dict[str, Any]:
        return {"type": self.type, "value": self.value}

    @staticmethod
    def from_value(value: Any) -> "Const":
        try:
            value_bytes = msgpack.packb(value)
            value_type = "msgpack"
        except Exception as e:
            _ = e
            value_bytes = cloudpickle.dumps(value)
            value_type = "cloudpickle"

        return Const(type=value_type, value=base64.b64encode(value_bytes).decode("ascii"))

    def to_value(self) -> Any:
        value_bytes = base64.b64decode(self.value.encode("ascii"))

        if self.type == "msgpack":
            return msgpack.unpackb(value_bytes)

        if self.type == "cloudpickle":
            return cloudpickle.loads(value_bytes)

        raise ValueError(f"Invalid type '{self.type}'")


@dataclass(frozen=True)
class ConstKey:
    key: str

    def __post_init__(self):
        assert re.match(_key_pattern, self.key), f"Key '{self.key}' must match pattern '{_key_pattern}'"

    def to_str(self) -> str:
        return f"consts:{self.key}"


@dataclass(frozen=True)
class InputKey:
    key: str

    def __post_init__(self):
        assert re.match(_key_pattern, self.key), f"Key '{self.key}' must match pattern '{_key_pattern}'"

    def to_str(self) -> str:
        return f"inputs:{self.key}"


@dataclass(frozen=True)
class NodeKey:
    key: str

    def __post_init__(self):
        assert re.match(_key_pattern, self.key), f"Key '{self.key}' must match pattern '{_key_pattern}'"

    def to_str(self) -> str:
        return f"nodes:{self.key}"


@dataclass(frozen=True)
class NodeInputKey:
    key: str
    input: str

    def __post_init__(self):
        assert re.match(_key_pattern, self.key), f"Key '{self.key}' must match pattern '{_key_pattern}'"
        assert re.match(_key_pattern, self.input), f"Input '{self.input}' must match pattern '{_key_pattern}'"

    def to_str(self) -> str:
        return f"nodes:{self.key}:{self.input}"


@dataclass(frozen=True)
class NodeOutputKey:
    key: str
    output: str

    def __post_init__(self):
        assert re.match(_key_pattern, self.key), f"Key '{self.key}' must match pattern '{_key_pattern}'"
        assert re.match(_key_pattern, self.output), f"Output '{self.output}' must match pattern '{_key_pattern}'"

    def to_str(self) -> str:
        return f"nodes:{self.key}:{self.output}"


@dataclass(frozen=True)
class OutputKey:
    key: str

    def __post_init__(self):
        assert re.match(_key_pattern, self.key), f"Key '{self.key}' must match pattern '{_key_pattern}'"

    def to_str(self) -> str:
        return f"outputs:{self.key}"


def _get_key_from_str(key: str) -> Union[ConstKey, InputKey, NodeOutputKey]:
    if not isinstance(key, str):
        raise TypeError(f"key must be a string; got type '{type(key)}'")

    splits = key.split(":")
    if len(splits) == 1:
        raise ValueError(f"key must have a prefix delimited by a semicolon: '{key}'")

    prefix = splits.pop(0)
    if prefix == "consts":
        return ConstKey(*splits)
    elif prefix == "inputs":
        return InputKey(*splits)
    elif prefix == "nodes":
        return NodeOutputKey(*splits)

    raise ValueError(f"invalid key: '{key}'")


@dataclass(frozen=True)
class FunctionCall:
    function: Union[Callable, str]
    args: Dict[str, Union[ConstKey, InputKey, NodeOutputKey]]

    def __post_init__(self):
        assert isinstance(self.function, str) or callable(
            self.function
        ), f"Function must be a string or callable; got type '{type(self.function)}'"
        assert isinstance(self.args, dict), f"Args must be a dictionary; got type '{type(self.args)}'"

        for param, arg in self.args.items():
            assert re.match(_key_pattern, param), f"Parameter '{param}' must match pattern '{_key_pattern}'"
            assert isinstance(
                arg, (ConstKey, InputKey, NodeOutputKey)
            ), f"Arg '{arg}' must ConstKey, InputKey, or NodeOutputKey; got type '{type(arg)}'"

    @staticmethod
    def from_dict(data: Dict) -> "FunctionCall":
        data = data.copy()
        function = data.pop("function")
        return FunctionCall(
            function=cloudpickle.loads(base64.b64decode(function.encode("ascii")))
            if data.pop("serialized", False)
            else function,
            args={arg: _get_key_from_str(key_str) for arg, key_str in data.pop("args").items()},
            **data,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "function": base64.b64encode(cloudpickle.dumps(self.function)).decode("ascii")
            if callable(self.function)
            else self.function,
            "serialized": callable(self.function),
            "args": {arg: key.to_str() for arg, key in self.args.items()},
        }


@dataclass(frozen=True)
class GraphCall:
    graph: "Graph"
    args: Dict[str, Union[ConstKey, InputKey, NodeOutputKey]]
    graph_name: Optional[str] = None

    def __post_init__(self):
        assert isinstance(self.graph, Graph), f"Graph must be a Graph; got type '{type(self.graph)}'"
        assert isinstance(self.args, dict), f"Args must be a dictionary; got type '{type(self.args)}'"
        assert self.graph_name is None or isinstance(
            self.graph_name, str
        ), f"Graph name must be a string or None; got type '{type(self.graph_name)}'"

        for param, arg in self.args.items():
            assert re.match(_key_pattern, param), f"Parameter '{param}' must match pattern '{_key_pattern}'"
            assert isinstance(
                arg, (ConstKey, InputKey, NodeOutputKey)
            ), f"Arg '{arg}' must ConstKey, InputKey, or NodeOutputKey; got type '{type(arg)}'"

    @staticmethod
    def from_dict(data: Dict) -> "GraphCall":
        data = data.copy()
        return GraphCall(
            graph=Graph.from_dict(data.pop("graph")),
            args={arg: _get_key_from_str(key_str) for arg, key_str in data.pop("args").items()},
            **data,
        )

    def to_dict(self) -> Dict[str, Any]:
        dct = {"graph": self.graph.to_dict(), "args": {arg: key.to_str() for arg, key in self.args.items()}}
        if self.graph_name is not None:
            dct["graph_name"] = self.graph_name
        return dct


@dataclass(frozen=True)
class Graph:
    consts: Dict[ConstKey, Const]
    inputs: Dict[InputKey, Optional[ConstKey]]
    nodes: Dict[NodeKey, Union[FunctionCall, GraphCall]]
    outputs: Dict[OutputKey, Union[ConstKey, InputKey, NodeOutputKey]]

    def __post_init__(self):
        assert isinstance(self.consts, dict), f"Consts must be a dictionary; got type '{type(self.consts)}'"
        assert isinstance(self.inputs, dict), f"Inputs must be a dictionary; got type '{type(self.inputs)}'"
        assert isinstance(self.nodes, dict), f"Nodes must be a dictionary; got type '{type(self.nodes)}'"
        assert isinstance(self.outputs, dict), f"Outputs must be a dictionary; got type '{type(self.outputs)}'"

        for key, const in self.consts.items():
            assert isinstance(key, ConstKey), f"Const key '{key}' must be type '{ConstKey}'"
            assert isinstance(const, Const), f"Const '{const}' must be type '{Const}'"

        for key, input_node in self.inputs.items():
            assert isinstance(key, InputKey), f"Input key '{key}' must be type '{InputKey}'"
            assert input_node is None or isinstance(
                input_node, ConstKey
            ), f"Input node '{input_node}' must be type '{ConstKey}' or None"

        for key, node in self.nodes.items():
            assert isinstance(key, NodeKey), f"Node key '{key}' must be type '{NodeKey}'"
            assert isinstance(
                node, (FunctionCall, GraphCall)
            ), f"Node '{node}' must be type '{FunctionCall}' or '{GraphCall}'"

        for key, output in self.outputs.items():
            assert isinstance(key, OutputKey), f"Output key '{key}' must be type '{OutputKey}'"
            assert isinstance(
                output, (ConstKey, InputKey, NodeOutputKey)
            ), f"Output '{output}' must be type '{ConstKey}', '{InputKey}', or '{NodeOutputKey}'"

    @staticmethod
    def from_dict(data: Dict) -> "Graph":
        """
        Create graph from graph dict by inferring the graph dict type

        :param data: graph dict
        :return: graph
        """
        if "edges" in data:
            return Graph.from_dict_with_edge_list(data)

        return Graph.from_dict_with_node_arguments(data)

    @staticmethod
    def from_dict_with_edge_list(data: Dict) -> "Graph":
        """
        Create graph from graph dict with edge list

        :param data: graph dict with edge list

            Example graph:

            .. code-block:: json

                {
                    "consts": {
                        "1": 1,
                        "2": {
                            "type": "int",
                            "value": "2"
                        }
                    },
                    "inputs: {
                        "b": "consts:1"
                    },
                    "nodes": {
                        "foo": {
                            "function": "foo"
                        },
                        "bar": {
                            "graph": "bar.json"
                        }
                    },
                    "edges": [
                        [
                            "consts:1",
                            "nodes:foo:inputs:a"
                        ],
                        [
                            "consts:2",
                            "nodes:foo:inputs:b"
                        ],
                        [
                            "nodes:foo:outputs:output",
                            "nodes:bar:inputs:a"
                        ],
                        [
                            "inputs:b",
                            "nodes:bar:inputs:b"
                        ]
                    ],
                    "outputs": {
                        "output": "nodes:bar:outputs:output"
                    }
                }

        :return: graph
        """
        # Use JSON Schema to produce better error messages
        jsonschema.validate(data, _graph_json_schema)

        data = copy.deepcopy(data)
        edges = data.pop("edges")
        nodes = data["nodes"]
        outputs = data["outputs"]

        for node in nodes.values():
            node["args"] = {}

        for src, dst in edges:
            src_splits = src.split(":")
            src_prefix = src_splits.pop(0)
            if src_prefix == "nodes":
                if len(src_splits) != 3 or src_splits[1] != "outputs":
                    raise ValueError(f"src key '{src}' is not a valid node output")
                new_src = f"nodes:{src_splits[0]}:{src_splits[2]}"
            else:
                new_src = src

            dst_splits = dst.split(":")
            dst_prefix = dst_splits.pop(0)
            if dst_prefix != "nodes" or len(dst_splits) != 3 or dst_splits[1] != "inputs":
                raise ValueError(f"dst key '{dst}' is not a valid node input")

            key = dst_splits[0]
            param = dst_splits[2]

            if key not in nodes:
                raise KeyError(f"node '{key}' does not exist")

            if param in nodes[key]["args"]:
                raise ValueError(f"parameter '{param}' in node '{key}' is already bound")

            nodes[key]["args"][param] = new_src

        for key, output in outputs.items():
            splits = output.split(":")
            prefix = splits.pop(0)
            if prefix == "nodes":
                if len(splits) != 3 or splits[1] != "outputs":
                    raise ValueError(f"output key '{output}' is not a valid node output")
                new_output = f"nodes:{splits[0]}:{splits[2]}"
            else:
                new_output = output

            outputs[key] = new_output

        return Graph.from_dict_with_node_arguments(data)

    @staticmethod
    def from_dict_with_node_arguments(data: Dict) -> "Graph":
        """
        Create graph from graph dict with node arguments

        :param data: graph dict with node arguments

            Example graph:

            .. code-block:: json

                {
                    "consts": {
                        "1": 1,
                        "2": {
                            "type": "int",
                            "value": "2"
                        }
                    },
                    "inputs: {
                        "b": "consts:1"
                    },
                    "nodes": {
                        "foo": {
                            "function": "foo",
                            "args": {
                                "a": "consts:1",
                                "b": "consts:2"
                            }
                        },
                        "bar": {
                            "graph": "bar.json",
                            "args": {
                                "a": "nodes:foo:output",
                                "b": "inputs:b"
                            }
                        }
                    },
                    "outputs": {
                        "output": "nodes:bar:output"
                    }
                }

        :return: graph
        """

        def _graph_node_from_dict(data: Union[Dict, str]) -> Union[FunctionCall, "GraphCall"]:
            if isinstance(data, dict) and "function" in data:
                return FunctionCall.from_dict(data)
            elif isinstance(data, dict) and "graph" in data:
                return GraphCall.from_dict(data)

            raise ValueError(f"invalid graph node dict '{data}'")

        data = data.copy()
        return Graph(
            consts={ConstKey(key=key): Const.from_dict(value) for key, value in data.pop("consts").items()},
            inputs={
                InputKey(key=key): _get_key_from_str(value) if value is not None else None
                for key, value in data.pop("inputs").items()
            },
            nodes={NodeKey(key=key): _graph_node_from_dict(value) for key, value in data.pop("nodes").items()},
            outputs={OutputKey(key=key): _get_key_from_str(value) for key, value in data.pop("outputs").items()},
            **data,
        )

    def to_dict(self) -> Dict[str, Any]:
        """
        Create graph representation to dictionary
        """
        graph_dict = {"consts": {}, "inputs": {}, "nodes": {}, "edges": [], "outputs": {}}

        for const_node_key, const_node in self.consts.items():
            graph_dict["consts"][const_node_key.key] = const_node.to_dict()

        for input_node_key, input_node in self.inputs.items():
            graph_dict["inputs"][input_node_key.key] = input_node.to_str() if input_node is not None else None

        for func_node_key, func_node in self.nodes.items():
            func_node_dict = func_node.to_dict()
            func_node_dict.pop("args")

            graph_dict["nodes"][func_node_key.key] = func_node_dict
            graph_dict["edges"].extend(
                [
                    [
                        arg.to_str() if not isinstance(arg, NodeOutputKey) else f"nodes:{arg.key}:outputs:{arg.output}",
                        f"nodes:{func_node_key.key}:inputs:{name}",
                    ]
                    for name, arg in func_node.args.items()
                ]
            )

        for output_node_key, output_node in self.outputs.items():
            graph_dict["outputs"][output_node_key.key] = (
                output_node.to_str()
                if not isinstance(output_node, NodeOutputKey)
                else f"nodes:{output_node.key}:outputs:{output_node.output}"
            )

        # Sanity check the final graph dictionary
        jsonschema.validate(graph_dict, _graph_json_schema)

        return graph_dict

    def to_dask(self, *args, **kwargs) -> Tuple[Dict[str, Any], List[str]]:
        inputs = {**dict(zip(self.inputs.keys(), args)), **kwargs}
        return self._convert_graph_to_dask_graph(inputs=inputs)

    def to_dot(
        self,
        rankdir: Literal["LR", "RL", "TB", "BT"] = "LR",
        no_input: bool = False,
        no_const: bool = False,
        no_output: bool = False,
    ) -> pydot.Dot:
        """
        Generate a dot graph from a graph

        :param rankdir: rank direction ("LR", "RL", "TB", "BT")
        :param no_input: do not include input nodes
        :param no_const: do not include const nodes
        :param no_output: do not include output nodes
        :return: dot graph
        """
        dot = pydot.Dot(rankdir=rankdir)

        for node_key, node in self.nodes.items():
            if isinstance(node, FunctionCall):
                input_names = tuple(inspect.signature(node.function).parameters.keys())
                output_names = _get_output_names(node.function)
                output_names = output_names if isinstance(output_names, tuple) else (output_names,)

                inputs = "|".join(f"<inputs_{key}>{key}" for key in input_names)
                outputs = "|".join(f"<outputs_{key}>{key}" for key in output_names)
                dot.add_node(
                    pydot.Node(
                        node_key.key, shape="record", label=f"{{{{{inputs}}}|{node.function.__name__}|{{{outputs}}}}}"
                    )
                )

            elif isinstance(node, GraphCall):
                inputs = "|".join(
                    f"<inputs_{arg_key}>{arg_key}"
                    for arg_key in sorted(
                        set(
                            arg.key
                            for subnode in node.graph.nodes.values()
                            for arg in subnode.args.values()
                            if isinstance(arg, InputKey)
                        )
                    )
                )
                outputs = "|".join(
                    f"<outputs_{output_key_str}>{output_key_str}"
                    for output_key_str in sorted(output_key.key for output_key in node.graph.outputs.keys())
                )
                graph_name = node.graph_name if node.graph_name is not None else node_key.key
                dot.add_node(
                    pydot.Node(node_key.key, shape="record", label=f"{{{{{inputs}}}|{graph_name}|{{{outputs}}}}}")
                )

            else:
                assert False, f"invalid node type '{type(node)}'"

            for param, arg in node.args.items():
                if isinstance(arg, NodeOutputKey):
                    dot.add_edge(
                        self._create_dot_edge(f"{arg.key}:outputs_{arg.output}", f"{node_key.key}:inputs_{param}")
                    )

                elif isinstance(arg, ConstKey):
                    if no_const:
                        continue

                    const = self.consts[arg]
                    value = const.to_value()

                    if isinstance(value, (int, float, bool, str)) or value is None:
                        label = str(value)
                    else:
                        label = type(value).__name__

                    dot.add_node(pydot.Node(f"const_{arg.key}", shape="box", label=label))
                    dot.add_edge(self._create_dot_edge(f"const_{arg.key}", f"{node_key.key}:inputs_{param}"))

                elif isinstance(arg, InputKey):
                    if no_input:
                        continue

                    dot.add_node(pydot.Node(f"input_{arg.key}", shape="parallelogram", label=arg.key))
                    dot.add_edge(self._create_dot_edge(f"input_{arg.key}", f"{node_key.key}:inputs_{param}"))

                else:
                    assert False, f"invalid path type '{type(arg)}'"

        for output_key, output in self.outputs.items():
            if no_output:
                continue

            output_uuid = f"_{uuid.uuid4().hex}"
            dot.add_node(pydot.Node(output_uuid, shape="ellipse", label=output_key.key))
            dot.add_edge(self._create_dot_edge(f"{output.key}:outputs_{output.output}", output_uuid))

        return dot

    @staticmethod
    def _create_dot_edge(src: str, dst: str) -> pydot.Edge:
        """
        This addresses a long-running bug in pydot where strings with colons are not properly quoted

        :param src: src string
        :param dst: dst string
        :return: pydot edge
        """
        edge = pydot.Edge(src, dst)

        if ":" in src:
            key, rest = src.split(":", maxsplit=1)
            edge.obj_dict["points"] = (f'"{key}":"{rest}"', edge.obj_dict["points"][1])

        if ":" in dst:
            key, rest = dst.split(":", maxsplit=1)
            edge.obj_dict["points"] = (edge.obj_dict["points"][0], f'"{key}":"{rest}"')

        return edge

    def _convert_graph_to_dask_graph(
        self,
        inputs: Optional[Dict[str, Any]] = None,
        input_mapping: Optional[Dict[InputKey, str]] = None,
        output_mapping: Optional[Dict[OutputKey, str]] = None,
    ) -> Tuple[Dict[str, Any], List[str]]:
        """
        Convert our own graph format to a dask graph.

        :param graph: graph to convert to dask graph
        :param inputs: inputs dictionary
        :param input_mapping: input mapping for subgraphs
        :param output_mapping: output mapping for subgraphs
        :return: tuple containing dask graph and targets
        """
        assert inputs is None or input_mapping is None, "cannot specify both inputs and input_mapping"

        dask_graph: dict = {}
        key_to_uuid: dict = {}

        # create constants
        for const_key, const in self.consts.items():
            graph_key = uuid.uuid4().hex
            dask_graph[graph_key] = const.to_value()
            key_to_uuid[const_key] = graph_key

        # create inputs
        if inputs is not None:
            for input_key in self.inputs.keys():
                graph_key = uuid.uuid4().hex
                dask_graph[graph_key] = inputs[input_key.key]
                key_to_uuid[input_key] = graph_key

        # assign random keys to all node paths and node output paths beforehand
        for node_uuid, node in self.nodes.items():
            if isinstance(node, FunctionCall):
                output_names = _get_output_names(node.function)
                output_names = output_names if isinstance(output_names, tuple) else (output_names,)
                for output_name in output_names:
                    key_to_uuid[NodeOutputKey(key=node_uuid.key, output=output_name)] = uuid.uuid4().hex

            elif isinstance(node, GraphCall):
                for output_key in node.graph.outputs.keys():
                    key_to_uuid[NodeOutputKey(key=node_uuid.key, output=output_key.key)] = uuid.uuid4().hex

        # overwrite keys of exported node output paths
        if output_mapping is not None:
            for output_name, graph_key in output_mapping.items():
                key_to_uuid[self.outputs[output_name]] = graph_key

        # import input mappings into current input mapping
        if input_mapping is not None:
            for input_key, const_path in self.inputs.items():
                if input_key in input_mapping:
                    key_to_uuid[input_key] = input_mapping[input_key]
                else:
                    key_to_uuid[input_key] = key_to_uuid[const_path]

        # build dask graph
        for node_key, node in self.nodes.items():
            if isinstance(node, FunctionCall):
                function_annotation = inspect.signature(node.function)

                # convert variadic positional arguments to positional arguments
                if next(iter(function_annotation.parameters.values())).kind == inspect.Parameter.VAR_POSITIONAL:
                    args = [key_to_uuid[path] for path in node.args.values()]

                # convert keyword arguments to positional arguments
                else:
                    args = []
                    for param_name, input_annotation in function_annotation.parameters.items():
                        # handle default arguments
                        if param_name not in node.args:
                            key = uuid.uuid4().hex
                            dask_graph[key] = input_annotation.default
                            args.append(key)
                            continue

                        path = node.args[param_name]
                        args.append(key_to_uuid[path])

                # unpack tuple output
                output_names = _get_output_names(node.function)
                node_uuid = uuid.uuid4().hex
                for output_position, output_name in (
                    enumerate(output_names) if isinstance(output_names, tuple) else ((None, output_names),)
                ):
                    graph_key = key_to_uuid[NodeOutputKey(key=node_key.key, output=output_name)]

                    if output_position is None:
                        node_uuid = graph_key
                        break

                    constant_key = uuid.uuid4().hex
                    dask_graph[constant_key] = output_position
                    dask_graph[graph_key] = (_unpack_tuple, node_uuid, constant_key)

                dask_graph[node_uuid] = (node.function,) + tuple(args)

            elif isinstance(node, GraphCall):
                new_input_mapping = {
                    InputKey(key=param_name): key_to_uuid[src_key] for param_name, src_key in node.args.items()
                }
                new_output_mapping = {
                    output_key: key_to_uuid[NodeOutputKey(key=node_key.key, output=output_key.key)]
                    for output_key in node.graph.outputs
                }
                dask_subgraph, _ = node.graph._convert_graph_to_dask_graph(
                    input_mapping=new_input_mapping, output_mapping=new_output_mapping
                )
                dask_graph.update(dask_subgraph)

        return dask_graph, [key_to_uuid[output_path] for output_path in self.outputs.values()]


def _unpack_tuple(graph_key: tuple, index: int) -> Any:
    return graph_key[index]
