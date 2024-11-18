import abc
import itertools
import logging
from collections import defaultdict, deque
from concurrent.futures import FIRST_COMPLETED, Future, wait
from typing import Any, DefaultDict, Dict, Hashable, Iterable, Optional, Set

from loky import get_reusable_executor

try:
    from pargraph.utility.graphlib_graphblas import TopologicalSorter
except ImportError:
    from graphlib import TopologicalSorter  # type: ignore[assignment,no-redef]


class Backend(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def submit(self, fn, /, *args, **kwargs) -> Future: ...


class GraphEngine:
    def __init__(self, backend: Optional[Backend] = None):
        """
        Graph Engine

        :param backend: parallel backend to use, if None, ProcessPoolExecutor is used
        """
        if backend is None:
            logging.info("parallel backend is not specified, using Loky's ProcessPoolExecutor as default")
            self.backend = get_reusable_executor()
        else:
            self.backend = backend

    def set_parallel_backend(self, backend: Backend) -> None:
        """
        Set parallel backend

        :param backend: parallel backend to use
        """
        self.backend = backend

    def get(self, graph: Dict, keys: Any, **kwargs) -> Any:
        """
        Compute dict graph

        :param graph: dict graph
        :param keys: keys to compute (e.g. ``"x"``, ``["x", "y", "z"]``, etc)
        :param kwargs: keyword arguments to forward to the parallel backend
        :return: results in the same structure as keys
        """
        keyset = set(self._flatten_iter([keys]))

        # cull graph to remove any unnecessary dependencies
        graphlib_graph = self._cull_graph(self._get_graph_dependencies(graph), keyset)
        ref_count_graph = self._create_ref_count_graph(graphlib_graph)

        topological_sorter = TopologicalSorter(graphlib_graph)
        topological_sorter.prepare()

        results: Dict[Hashable, Any] = {}
        future_to_key: Dict[Future[Any], Hashable] = {}

        # perform automatic reference counting to free results that are no longer needed
        def dereference_key(key):
            if key not in graphlib_graph:
                return

            for predecessor_key in graphlib_graph[key]:
                if predecessor_key not in ref_count_graph:
                    continue

                ref_count_graph[predecessor_key] -= 1

                # if reference count reaches 0, consider freeing the result
                if ref_count_graph[predecessor_key] == 0 and predecessor_key not in keyset:
                    results.pop(predecessor_key, None)

        # wait until at least one future gets resolved and handle it accordingly
        def wait_for_completed_futures():
            done, not_done = wait(future_to_key.keys(), return_when=FIRST_COMPLETED)
            done_keys = []

            for done_future in done:
                exception = done_future.exception()
                if exception is not None:
                    # cancelling futures may or may not work depending on implementation
                    for not_done_future in not_done:
                        not_done_future.cancel()
                    raise exception

                key = future_to_key[done_future]
                result = done_future.result()
                results[key] = result

                future_to_key.pop(done_future, None)
                done_keys.append(key)

            topological_sorter.done(*done_keys)
            for done_key in done_keys:
                dereference_key(done_key)

        # while there are still unscheduled tasks
        while topological_sorter.is_active():
            # get in vertices
            in_keys = topological_sorter.get_ready()

            # if there are no in-vertices, wait for a future to resolve
            # IMPORTANT: we make the assumption that the graph is acyclic
            if not in_keys:
                wait_for_completed_futures()
                continue

            for in_key in in_keys:
                computation = graph[in_key]

                if self._is_submittable_function_computation(computation):
                    future = self._submit_function_computation(computation, results, **kwargs)
                    future_to_key[future] = in_key
                else:
                    result = self._evaluate_computation(computation, results)
                    results[in_key] = result
                    topological_sorter.done(in_key)
                    dereference_key(in_key)

        # resolve all pending futures
        while future_to_key:
            wait_for_completed_futures()

        return self._pack_results(results, keys)

    @classmethod
    def _pack_results(cls, results: Dict, keys: Any) -> Any:
        keys_type = type(keys)
        if keys_type in (tuple, list, set, frozenset):
            return keys_type([cls._pack_results(results, key) for key in keys])
        elif keys_type is dict:
            return {k: cls._pack_results(results, v) for k, v in keys.items()}
        return results[keys]

    @classmethod
    def _flatten_iter(cls, seq: Iterable, container=list) -> Iterable:
        for el in seq:
            # recursively flatten specific container types
            if isinstance(el, container):
                yield from cls._flatten_iter(el, container=container)
            else:
                yield el

    @staticmethod
    def _is_submittable_function_computation(computation: Any) -> bool:
        return isinstance(computation, tuple) and len(computation) > 0 and callable(computation[0])

    def _submit_function_computation(self, computation: Any, results: Dict, **kwargs) -> Optional[Future]:
        func, *args = computation
        # implementation detail: arguments are evaluated eagerly on the main thread
        evaluated_args = iter(self._evaluate_computation(arg, results) for arg in args)
        return self.backend.submit(func, *evaluated_args, **kwargs)

    @classmethod
    def _evaluate_computation(cls, computation: Any, results: Dict) -> Optional[Any]:
        """
        Evaluate computation eagerly

        :param computation: computation to evaluate
        :param results: existing results
        :return: result
        """
        if isinstance(computation, tuple) and computation and callable(computation[0]):
            func, *args = computation
            return func(*iter(cls._evaluate_computation(arg, results) for arg in args))

        if isinstance(computation, list):
            return [cls._evaluate_computation(el, results) for el in computation]

        try:
            hash(computation)
        except TypeError:
            return computation

        if computation in results:
            return results[computation]

        return computation

    @staticmethod
    def _get_graph_dependencies(graph: Dict) -> Dict:
        keys = set(graph.keys())

        def flatten(value: Any) -> Set[Any]:
            # handle tasks as tuples
            if isinstance(value, tuple) and value and callable(value[0]):
                sets = [flatten(sub_value) for sub_value in itertools.islice(value, 1, None)]
                return set.union(*sets) if sets else set()

            # handle list of values
            if isinstance(value, list):
                sets = [flatten(sub_value) for sub_value in value]
                return set.union(*sets) if sets else set()

            # check if value is hashable
            try:
                hash(value)
            except TypeError:
                return set()

            # check if value is an existing key
            if value in keys:
                return {value}

            return set()

        return {key: flatten(value) for key, value in graph.items()}

    @staticmethod
    def _create_ref_count_graph(graph: Dict) -> Dict:
        ref_count_graph: DefaultDict[str, int] = defaultdict(int)
        for value in graph.values():
            for sub_value in value:
                ref_count_graph[sub_value] += 1
        return ref_count_graph

    @staticmethod
    def _cull_graph(graph: Dict, target_keys: Set) -> Dict:
        """
        Cull graph by performing BFS on target keys
        """
        queue = deque(target_keys)
        visited = set()
        for target_key in target_keys:
            visited.add(target_key)

        while queue:
            key = queue.popleft()

            for predecessor_key in graph[key]:
                if predecessor_key in visited:
                    continue
                visited.add(predecessor_key)
                queue.append(predecessor_key)

        return {key: graph[key] for key in visited}
