from typing import Callable
from collections import OrderedDict
from textwrap import dedent, indent
from abc import ABCMeta
import inspect


def get_method_meta(func: Callable):
    """
    Returns a metaclass that defines a class as a method for the specified function ``func``.
    Any class that is defined with this metaclass will be registered as a method for the specified function.
    Its docstring will be added to the function's docstring.
    """

    additional_docstr_format = """
    Methods
    -------
    method="{name}()"

        .. code-block:: python

            {name}{signature}
    """
    class DefMethodMeta(ABCMeta):
        """
        Define a meta class where the class is registered as a method for the specified function ``func``.
        """
        def __init__(cls, name, bases, dct):
            super().__init__(name, bases, dct)
            if bases:
                # register the class as a method for the `seq1d` function
                setattr(func, name, cls)
                if not hasattr(func, "methods"):
                    setattr(func, "methods", [])
                setattr(func, "methods", getattr(func, "methods") + [name])

                # add func's docstr
                # get the signature of the cls
                signature = inspect.signature(cls.__init__)
                new_params = OrderedDict(
                    (nm, param) for nm, param in signature.parameters.items() if nm != 'self'
                )
                signature = signature.replace(parameters=new_params.values())
                func.__doc__ += additional_docstr_format.format(name=name, signature=signature)
                func.__doc__ += indent(dedent(cls.__doc__), " " * 8)
            else:
                # cls is the base method class
                setattr(func, "base_method", cls)

    return DefMethodMeta

def check_method(method, func_obj: Callable):
    # check if the passed ``method`` for a function ``func_obj`` is a valid method
    # func_obj must be the ones passed in get_method_meta above
    if not isinstance(method, func_obj.base_method):
        msg = f"`method` must be an instance of `{func_obj.__name__}.*` method, got {type(method)}. "
        msg += f"Available methods are: {func_obj.methods}"
        raise ValueError(msg)
