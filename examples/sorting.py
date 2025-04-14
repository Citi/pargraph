from typing import List

import numpy as np

from pargraph import GraphEngine, delayed, graph


@delayed
def filter_array(array: np.ndarray, low: float, high: float) -> np.ndarray:
    """
    Filter an array to include only values within the specified range.

    :param array: The input array to be filtered
    :param low: The lower bound of the range
    :param high: The upper bound of the range
    :return: A new array containing only the values within the specified range
    """
    return array[(array >= low) & (array <= high)]


@delayed
def sort_array(array: np.ndarray) -> np.ndarray:
    """
    Sort array

    :param array: The input array to be sorted
    :return: A new array containing the sorted values
    """
    return np.sort(array)


@delayed
def reduce_arrays(*arrays: np.ndarray) -> np.ndarray:
    """
    Reduce arrays by concatenating them

    :param arrays: A variable number of arrays to be concatenated
    :return: A new array containing all the concatenated values
    """
    return np.concatenate(arrays)


@graph
def map_reduce_sort(array: np.ndarray, partition_count: int) -> np.ndarray:
    """
    Map reduce sort

    :param array: The input array to be sorted
    :param partition_count: The number of partitions to divide the array into
    :return: A new array containing the sorted values
    """
    return reduce_arrays(
        *(
            sort_array(filter_array(array, i / partition_count, (i + 1) / partition_count))
            for i in range(partition_count)
        )
    )


@graph
def map_reduce_sort_recursive(
    array: np.ndarray, partition_counts: List[int], _low: float = 0, _high: float = 1
) -> np.ndarray:
    """
    Map reduce sort recursively

    :param array: The input array to be sorted
    :param partition_counts: A list of partition counts for recursive sorting
    :param _low: The lower bound of the values range (not supposed to be used publicly)
    :param _high: The upper bound of the values range (not supposed to be used publicly)
    :return: A new array containing the sorted values
    """
    if len(partition_counts) == 0:
        return sort_array(array)

    partition_count, *partition_counts = partition_counts

    sorted_partitions = []
    for i in range(partition_count):
        low = _low + (_high - _low) * (i / partition_count)
        high = _low + (_high - _low) * ((i + 1) / partition_count)
        sorted_partitions.append(map_reduce_sort_recursive(filter_array(array, low, high), partition_counts, low, high))

    return reduce_arrays(*sorted_partitions)


if __name__ == "__main__":
    task_graph, keys = map_reduce_sort_recursive.to_graph(partition_counts=[2, 2, 2]).to_dict(
        array=np.random.rand(1_000_000)
    )

    graph_engine = GraphEngine()
    print(graph_engine.get(task_graph, keys)[0])
