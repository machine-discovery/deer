from typing import Any, List, Tuple, Callable, Sequence, Optional

import equinox as eqx
import jax
import jax.numpy as jnp
from jax._src import prng

from deer.fseq1d import seq1d


def vmap_to_shape(func: Callable, shape: Sequence[int]) -> Callable:
    rank = len(shape)
    for i in range(rank - 1):
        func = jax.vmap(func)
    return func


def custom_mlp(mlp: eqx.nn.MLP, key: prng.PRNGKeyArray, init_method: Optional[str] = "he_uniform") -> eqx.nn.MLP:
    """
    eqx.nn.MLP with custom initialisation scheme using jax.nn.initializers
    """
    where_bias = lambda m: [lin.bias for lin in m.layers]
    where_weight = lambda m: [lin.weight for lin in m.layers]

    mlp = eqx.tree_at(where=where_bias, pytree=mlp, replace_fn=jnp.zeros_like)

    if init_method is None:
        return mlp

    if init_method == "he_uniform":
        # get all the weights of the mlp model
        weights = where_weight(mlp)
        # split the random key into different subkeys for each layer
        subkeys = jax.random.split(key, len(weights))
        new_weights = [
            jax.nn.initializers.he_uniform()(subkey, weight.shape) for weight, subkey in zip(weights, subkeys)
        ]
        mlp = eqx.tree_at(where=where_weight, pytree=mlp, replace=new_weights)
    else:
        return NotImplementedError("only he_uniform is implemented")
    return mlp


def custom_linear(linear: eqx.nn.Linear, key: prng.PRNGKeyArray, init_method: Optional[str] = "he_uniform", nstate: Optional[int] = None) -> eqx.nn.MLP:
    """
    eqx.nn.MLP with custom initialisation scheme using jax.nn.initializers
    """
    where_bias = lambda m: [m.bias]
    where_weight = lambda m: [m.weight]

    linear = eqx.tree_at(where=where_bias, pytree=linear, replace_fn=jnp.zeros_like)

    if init_method is None:
        return linear

    if init_method == "he_uniform":
        # get all the weights of the mlp model
        weights = where_weight(linear)
        # split the random key into different subkeys for each layer
        subkeys = jax.random.split(key, len(weights))
        new_weights = [
            jax.nn.initializers.he_uniform()(subkey, weight.shape) for weight, subkey in zip(weights, subkeys)
        ]
        linear = eqx.tree_at(where=where_weight, pytree=linear, replace=new_weights)
    elif init_method == "he_normal":
        # get all the weights of the mlp model
        weights = where_weight(linear)
        # split the random key into different subkeys for each layer
        subkeys = jax.random.split(key, len(weights))
        new_weights = [
            jax.nn.initializers.he_normal()(subkey, weight.shape) for weight, subkey in zip(weights, subkeys)
        ]
        linear = eqx.tree_at(where=where_weight, pytree=linear, replace=new_weights)
    elif init_method == "uniform":
        assert nstate is not None
        # get all the weights of the mlp model
        weights = where_weight(linear)
        # split the random key into different subkeys for each layer
        subkeys = jax.random.split(key, 2)
        new_weights = [jax.nn.initializers.uniform(scale=1 / jnp.sqrt(nstate))(subkeys[0], weights[0].shape)]
        linear = eqx.tree_at(where=where_weight, pytree=linear, replace=new_weights)

        bias = where_bias(linear)
        new_bias = [jax.nn.initializers.uniform(scale=1 / jnp.sqrt(nstate))(subkeys[1], bias[0].shape)]
        linear = eqx.tree_at(where=where_bias, pytree=linear, replace=new_bias)
    else:
        return NotImplementedError("only he_uniform is implemented")
    return linear


def custom_gru(gru: eqx.nn.GRUCell, key: prng.PRNGKeyArray) -> eqx.nn.GRUCell:
    """
    eqx.nn.GRUCell with custom initialisation scheme using jax.nn.initializers
    """
    where_bias = lambda g: g.bias
    where_bias_n = lambda g: g.bias_n
    where_weight_ih = lambda g: g.weight_ih
    where_weight_hh = lambda g: g.weight_hh

    gru = eqx.tree_at(where=where_bias, pytree=gru, replace_fn=jnp.zeros_like)
    gru = eqx.tree_at(where=where_bias_n, pytree=gru, replace_fn=jnp.zeros_like)

    weight_ih = where_weight_ih(gru)
    weight_hh = where_weight_hh(gru)

    ih_key, hh_key = jax.random.split(key, 2)

    new_weight_ih = jax.nn.initializers.lecun_normal()(ih_key, weight_ih.shape)
    new_weight_hh = jax.nn.initializers.orthogonal()(hh_key, weight_hh.shape)

    gru = eqx.tree_at(where_weight_ih, gru, new_weight_ih)
    gru = eqx.tree_at(where_weight_hh, gru, new_weight_hh)
    return gru


class MLP(eqx.Module):
    model: eqx.nn.MLP

    def __init__(self, ninp: int, nstate: int, nout: int, key: prng.PRNGKeyArray):
        self.model = eqx.nn.MLP(
            in_size=ninp,
            out_size=nout,
            width_size=nstate,
            depth=1,
            activation=jax.nn.relu,
            # final_activation=jax.nn.tanh,  # adding --> even smaller gradient
            key=key
        )
        self.model = custom_mlp(self.model, key)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        return vmap_to_shape(self.model, x.shape)(x)


class Linear(eqx.Module):
    model: eqx.nn.Linear

    def __init__(
        self, ninp: int,
        nout: int,
        key: prng.PRNGKeyArray,
        init_method: Optional[str] = "he_uniform",
    ):
        self.model = eqx.nn.Linear(
            in_features=ninp,
            out_features=nout,
            use_bias=True,
            key=key
        )
        self.model = custom_linear(self.model, key, init_method, nout)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        return vmap_to_shape(self.model, x.shape)(x)


class GRU(eqx.Module):
    gru: eqx.Module
    use_scan: bool

    def __init__(self, ninp: int, nstate: int, key: prng.PRNGKeyArray, use_scan: bool):
        self.gru = eqx.nn.GRUCell(
            input_size=ninp,
            hidden_size=nstate,
            key=key
        )
        self.gru = custom_gru(self.gru, key)
        self.use_scan = use_scan

    def __call__(self, inputs: jnp.ndarray, h0: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        # h0.shape == (nbatch, nstate)
        # inputs.shape == (nbatch, ninp)
        assert len(inputs.shape) == len(h0.shape)

        states = vmap_to_shape(self.gru, inputs.shape)(inputs, h0)
        return states, states


class SingleScaleGRU(eqx.Module):
    nchannel: int
    nlayer: int
    encoder: MLP
    grus: List[List[GRU]]
    mlps: List[MLP]
    classifier: MLP
    norms: List[eqx.nn.LayerNorm]
    dropout: eqx.nn.Dropout
    dropout_key: prng.PRNGKeyArray
    use_scan: bool

    def __init__(self, ninp: int, nchannel: int, nstate: int, nlayer: int, nclass: int, key: prng.PRNGKeyArray, use_scan: bool):
        keycount = 1 + (nchannel + 1) * nlayer + 1 + 1  # +1 for dropout
        print(f"Keycount: {keycount}")
        keys = jax.random.split(key, keycount)

        self.nchannel = nchannel
        self.nlayer = nlayer

        assert nstate % nchannel == 0
        gru_nstate = int(nstate / nchannel)

        # encode inputs (or rather, project) to have nstate in the feature dimension
        self.encoder = MLP(ninp=ninp, nstate=nstate, nout=nstate, key=keys[0])

        # nlayers of (scale_gru + mlp) pair
        self.grus = [[
            GRU(
                ninp=nstate,
                nstate=gru_nstate,
                key=keys[int(1 + (nchannel * j) + i)],
                use_scan=use_scan
            ) for i in range(nchannel)] for j in range(nlayer)
        ]
        self.mlps = [
            MLP(ninp=nstate, nstate=nstate, nout=nstate, key=keys[int(i + 1 + nchannel * nlayer)]) for i in range(nlayer)
        ]
        assert len(self.grus) == nlayer
        assert len(self.grus[0]) == nchannel
        print(f"scale_grus random keys end at index {int(1 + (nchannel * (nlayer - 1)) + (nchannel - 1))}")
        print(f"mlps random keys end at index {int((nchannel * nlayer) + nlayer)}")

        # project nstate in the feature dimension to nclasses for classification
        self.classifier = MLP(ninp=nstate, nstate=nstate, nout=nclass, key=keys[int((nchannel + 1) * nlayer + 1)])

        self.norms = [eqx.nn.LayerNorm((nstate,), use_weight=False, use_bias=False) for i in range(nlayer * 2)]
        self.dropout = eqx.nn.Dropout(p=0.2)
        self.dropout_key = keys[-1]

        self.use_scan = use_scan

    def __call__(self, inputs: jnp.ndarray, h0: jnp.ndarray, yinit_guess: jnp.ndarray) -> jnp.ndarray:
        # encode (or rather, project) the inputs
        inputs = self.encoder(inputs)

        def model_func(carry: jnp.ndarray, inputs: jnp.ndarray, model: Any):
            return model(inputs, carry)[1]  # could be [0] or [1]

        for i in range(self.nlayer):
            inputs = self.norms[i](inputs)

            x_from_all_channels = []

            for ch in range(self.nchannel):
                if self.use_scan:
                    model = lambda carry, inputs: self.grus[i][ch](inputs, carry)
                    x = jax.lax.scan(model, h0, inputs)[1]
                else:
                    x = seq1d(
                        model_func,
                        h0,
                        inputs,
                        self.grus[i][ch],
                        yinit_guess,
                    )
                x_from_all_channels.append(x)

            x = jnp.concatenate(x_from_all_channels, axis=-1)
            x = self.norms[i + 1](x + inputs)  # add and norm after multichannel GRU layer
            x = self.mlps[i](x) + x  # add with norm added in the next loop
            inputs = x
        return self.classifier(x)


class LEMCell(eqx.Module):
    dt: float
    inp2hid: eqx.nn.Linear
    hid2hid: eqx.nn.Linear
    transform_z: eqx.nn.Linear

    def __init__(self, ninp: int, nhid: int, dt: float, key: prng.PRNGKeyArray):
        keys = jax.random.split(key, 3)
        self.dt = dt
        self.inp2hid = Linear(ninp, 4 * nhid, key=keys[0], init_method="he_normal")
        self.hid2hid = Linear(nhid, 3 * nhid, key=keys[1], init_method="he_normal")
        self.transform_z = Linear(nhid, nhid, key=keys[2], init_method="he_normal")

    def __call__(self, x: jnp.ndarray, yz: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        y, z = jnp.split(yz, 2, axis=-1)
        transformed_inp = self.inp2hid(x)
        transformed_hid = self.hid2hid(y)
        i_dt1, i_dt2, i_z, i_y = jnp.split(transformed_inp, 4, axis=-1)
        h_dt1, h_dt2, h_y = jnp.split(transformed_hid, 3, axis=-1)

        # Equation 3 in the LEM paper
        ms_dt_bar = self.dt * jax.nn.sigmoid(i_dt1 + h_dt1)
        ms_dt = self.dt * jax.nn.sigmoid(i_dt2 + h_dt2)

        z = (1 - ms_dt) * z + ms_dt * jax.nn.tanh(i_y + h_y)
        y = (1 - ms_dt_bar) * y + ms_dt_bar * jax.nn.tanh(self.transform_z(z) + i_z)
        yz = jnp.concatenate((y, z), axis=-1)
        return yz


class LEM(eqx.Module):
    lem: eqx.Module
    use_scan: bool

    def __init__(self, ninp: int, nstate: int, dt: bool, key: prng.PRNGKeyArray, use_scan: bool):
        self.lem = LEMCell(
            ninp=ninp,
            nhid=nstate,
            dt=dt,
            key=key
        )
        # self.gru = custom_gru(self.gru, key)
        self.use_scan = use_scan

    def __call__(self, inputs: jnp.ndarray, yz0: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        # yz0.shape == (nbatch, nstate * 2)
        # inputs.shape == (nbatch, ninp)

        yzstates = vmap_to_shape(self.lem, inputs.shape)(inputs, yz0)
        return yzstates, yzstates


class ScaledLEM(eqx.Module):
    lems: LEM
    classifier: MLP
    use_scan: bool

    def __init__(
        self,
        ninp: int,
        nstate: int,
        nclass: int,
        key: prng.PRNGKeyArray,
        use_scan: bool,
        dt: float = 0.0016
    ):
        keycount = 2
        print(f"Keycount: {keycount}")
        keys = jax.random.split(key, keycount)

        self.lems = LEM(
            ninp=ninp,
            nstate=nstate,
            dt=dt,
            key=keys[0],
            use_scan=use_scan
        )

        # project nstate in the feature dimension to nclasses for classification
        self.classifier = Linear(ninp=nstate, nout=nclass, key=keys[1], init_method="he_normal")

        self.use_scan = use_scan

    def __call__(self, inputs: jnp.ndarray, yz0: jnp.ndarray, yinit_guess: jnp.ndarray) -> jnp.ndarray:
        # encode (or rather, project) the inputs

        def model_func(yzcarry: jnp.ndarray, inputs: jnp.ndarray, model: Any):
            return model(inputs, yzcarry)[1]

        if self.use_scan:
            model = lambda carry, inputs: self.lems(inputs, carry)
            x = jax.lax.scan(model, yz0, inputs)[1]
        else:
            x = seq1d(
                model_func,
                yz0,
                inputs,
                self.lems,
                yinit_guess,
            )
        x, _ = jnp.split(x, 2, axis=-1)

        return self.classifier(x)
