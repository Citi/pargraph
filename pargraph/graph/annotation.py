import inspect
import sys
from typing import Callable, Tuple, Union

if sys.version_info < (3, 9):
    from typing_extensions import Annotated, get_args, get_origin
else:
    from typing import Annotated, get_args, get_origin


class Result:
    """
    Annotate a function output with a name

    .. code:: python

        @delayed
        def my_function() -> Annotated[str, Result("my_output")]:
            return "Hello, World!"

    """

    def __init__(self, name: str):
        """
        Initialize Result annotation

        :param name: result name
        """
        self.name = name

    def get_name(self) -> str:
        """
        Get result name

        :return: result name
        """
        return self.name


def _get_output_names(function: Callable) -> Union[str, Tuple[str, ...]]:
    annotation = inspect.signature(function).return_annotation
    origin = get_origin(annotation)

    if origin is not tuple:
        if get_origin(annotation) is not Annotated:
            return "result"

        _, *meta = get_args(annotation)
        results = [m for m in meta if isinstance(m, Result)]
        if len(results) != 1:
            raise ValueError(f"output with type '{annotation}' must have exactly one Result annotation")

        return results[0].get_name()

    output_names = []

    for i, arg in enumerate(get_args(annotation)):
        if get_origin(arg) is not Annotated:
            result_name = f"result_{i}"

            if result_name in output_names:
                raise ValueError(f"found duplicate output name '{result_name}'")

            output_names.append(result_name)
            continue

        _, *meta = get_args(arg)
        results = [m for m in meta if isinstance(m, Result)]
        if len(results) != 1:
            raise ValueError(f"output with type '{arg}' must have exactly one Result annotation")

        result_name = results[0].get_name()
        if result_name in output_names:
            raise ValueError(f"found duplicate output name '{result_name}'")

        output_names.append(result_name)

    return tuple(output_names)
