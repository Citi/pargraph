from pargraph import GraphEngine, graph


@graph
def pythagoras(a: float, b: float) -> float:
    """
    Calculate the length of the hypotenuse of a right triangle given the lengths of the other two sides.

    :param a: Length of one side of the triangle.
    :param b: Length of the other side of the triangle.
    :return: Length of the hypotenuse.
    """
    return (a**2 + b**2) ** 0.5


if __name__ == "__main__":
    task_graph, keys = pythagoras.to_graph().to_dict(a=3, b=4)

    graph_engine = GraphEngine()
    print(graph_engine.get(task_graph, keys)[0])
