"""Contains classes and functions related to the dynamic import of objects."""

from typing import Any, Mapping

from nnfabrik.utility.nnf_helper import dynamic_import, split_module_name


def import_object(
    path: str,
    object_kwargs: Mapping[str, Any] = None,
    split_func=split_module_name,
    import_func=dynamic_import,
) -> Any:
    """Imports an object given a path, calls it with the given keyword arguments and returns the object returned by the
    imported object.

    Args:
        path: The absolute path to the object to be imported. For example: "module1.module2.class"
        object_kwargs: The keyword arguments used to call the imported object.
        split_func: Callable that, given a module's full path (absolute path + name), splits it into absolute path and
            name. Optional. Example: given "module1.module2.class", returns "module1.module2", "class".
        import_func: Callable that, given a module's path and name, returns the object associated to the module to be
            imported. Optional.

    Returns:
        The object returned by the call to the imported object.

    Raises:
        ValueError: Given path is not absolute or does not exist or imported object is not callable.
    """
    if object_kwargs is None:
        object_kwargs = dict()
    module_path, name = split_func(path)
    if not module_path:
        raise ValueError(f"'{path}' is not a absolute path")
    try:
        obj = import_func(module_path, name)
    except ModuleNotFoundError:
        raise ValueError(f"Module in '{path}' not found")
    except AttributeError:
        raise ValueError(f"Object in '{path}' not found")
    try:
        return obj(**object_kwargs)
    except TypeError:
        raise ValueError(f"Imported object from '{path}' is not callable")
