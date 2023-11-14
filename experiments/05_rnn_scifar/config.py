from typing import Dict, Any, Sequence, List, Tuple, Optional
import yaml
import jax
import optax
import equinox as eqx
from models import RNNNet
from cases import Case, SeqCIFAR10


optimizer_kwargs = {
    # clipping
    "clip_by_global_norm": {
        "class": optax.clip_by_global_norm,
        "kwargs": [
            ("max_norm", float, 1.0),
        ],
    },
    # optimizer
    "adam": {
        "class": optax.adam,
        "kwargs": [
            ("learning_rate", float, 1e-3),
            ("b1", float, 0.9),
            ("b2", float, 0.999),
            ("eps", float, 1e-8),
        ],
    },
    "adamw": {
        "class": optax.adamw,
        "kwargs": [
            ("learning_rate", float, 1e-3),
            ("b1", float, 0.9),
            ("b2", float, 0.999),
            ("eps", float, 1e-8),
            ("weight_decay", float, 1e-4),
        ],
    },
    # scheduler
    "warmup_cosine_decay_schedule": {
        "class": optax.warmup_cosine_decay_schedule,
        "kwargs": [
            ("init_value", float, 1e-5),
            ("peak_value", float, 6e-4),
            ("warmup_steps", int, 100_000),
            ("decay_steps", int, 150_000),
            ("end_value", float, 1e-6),
        ],
    },
}

model_kwargs = {
    "rnnnet": {
        "class": RNNNet,
        "kwargs": [
            ("nhiddens", int, 64),
            ("nlayers", int, 8),
            ("nhiddens_mlp", int, 64),
            ("num_heads", int, 1),
            ("method", str, "deer"),
            ("rnn_type", str, "gru"),
            ("bidirectional", bool, False),
            ("bidirasymm", bool, False),
            ("rnn_wrapper", int, None),
            ("p_dropout", float, None),
            ("prenorm", bool, False),
            ("final_mlp", bool, False),
        ],
    },
}

case_kwargs = {
    "scifar10": {
        "class": SeqCIFAR10,
        "kwargs": [
            ("rootdir", str, "data/cifar10"),
        ],
    },
}

def read_config(config_fname: str = "config.py") -> Dict[str, Dict[str, Any]]:
    with open(config_fname, "r") as fh:
        config = yaml.load(fh, Loader=yaml.FullLoader)
    return config

def get_optimizer_from_dct(opt_dct) -> optax.GradientTransformation:
    if isinstance(opt_dct, list):
        lst = [get_optimizer_from_dct(dct) for dct in opt_dct]
        return optax.chain(*lst)
    elif isinstance(opt_dct, dict):
        if "name" in opt_dct:
            name = opt_dct.pop("name")
            clss = optimizer_kwargs[name]["class"]
            avail_kwargs = optimizer_kwargs[name]["kwargs"]
            kwargs = {}
            for kwkey, kwtype, defval in avail_kwargs:
                if kwkey in opt_dct:
                    if isinstance(opt_dct[kwkey], dict):
                        kwargs[kwkey] = get_optimizer_from_dct(opt_dct.pop(kwkey))
                    else:
                        kwargs[kwkey] = kwtype(opt_dct.pop(kwkey))
                else:
                    kwargs[kwkey] = defval
            return clss(**kwargs)
        else:
            raise ValueError(f"Invalid dict: {opt_dct}")
    else:
        raise TypeError(f"Invalid type: {type(opt_dct)}")

def get_model_from_dct(model_dct: Dict[str, Any], num_inps: int, num_outs: int, with_embedding: bool,
                       reduce_length: bool, *, key: jax.random.PRNGKey) -> eqx.Module:
    return _get_obj_from_dct(model_dct, model_kwargs,
                             num_inps=num_inps, num_outs=num_outs, with_embedding=with_embedding,
                             reduce_length=reduce_length, key=key)

def get_case_from_dct(case_dct: Dict[str, Any]) -> Case:
    return _get_obj_from_dct(case_dct, case_kwargs)

def _get_obj_from_dct(dct: Optional[Dict[str, Any]], which_kwargs: Dict[str, Any], *args, **kwargs0):
    if dct is None:
        return None
    name = dct.pop("name")

    # get the kwargs
    details = which_kwargs[name]
    kwargs = _get_kwargs(details["kwargs"], dct)

    # get the object
    return details["class"](*args, **kwargs, **kwargs0)

def _get_kwargs(all_kwargs: List[Tuple[str, type, Any]], dct: Dict[str, Any]) -> Dict[str, Any]:
    kwargs = {}
    for argname, argtype, defval in all_kwargs:
        if argname in dct:
            kwargs[argname] = argtype(dct.pop(argname))
        else:
            kwargs[argname] = defval

    if len(dct) > 0:
        raise ValueError(f"Unknown arguments: {dct.keys()}")

    return kwargs
