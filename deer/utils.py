from typing import Callable, Union
import jax
import jax.numpy as jnp
from collections import OrderedDict
from textwrap import dedent, indent
from abc import ABCMeta
import inspect


class Result:
    """
    An object to store the result of the iterative algorithm.
    """
    value: jnp.ndarray
    success: jnp.ndarray  # in bool with the same shape as value

    def __init__(self, value: jnp.ndarray, success: Union[bool, None, jnp.ndarray] = None):
        self.value = value
        if success is None:
            success = jnp.full_like(value, True, dtype=jnp.bool)
        elif isinstance(success, bool):
            success = jnp.full_like(value, success, dtype=jnp.bool)
        elif isinstance(success, jnp.ndarray):
            assert success.dtype == jnp.bool
            success = jnp.broadcast_to(success, value.shape)
        # no else with type error because sometimes the JAX tracer put strings of the type in the inputs
        # TODO: think how to handle this
        self.success = success

jax.tree_util.register_pytree_node(
    Result,
    (lambda res: ((res.value, res.success), None)),  # flatten
    (lambda aux_data, children: Result(*children)),  # unflatten
)

def get_method_meta(func: Callable):
    """
    Returns a metaclass that defines a class as a method for the specified function ``func``.
    Any class that is defined with this metaclass will be registered as a method for the specified function.
    Its docstring will be added to the function's docstring.
    """

    additional_docstr_format = """
    Methods
    -------
    method={funcname}.{name}()

        .. code-block:: python

            {funcname}.{name}{signature}
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
                func.__doc__ += additional_docstr_format.format(name=name, signature=signature, funcname=func.__name__)
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
