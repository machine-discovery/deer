from typing import Tuple, Dict, Any, Optional
import os
from functools import partial
import argparse
import pickle
from tqdm import tqdm
import jax
import jax.numpy as jnp
from jax.experimental.ode import odeint
import matplotlib.pyplot as plt
from tensorboardX import SummaryWriter
from flax import linen as nn
import optax
import numpy as np
import scipy.integrate
from deer.seq1d import solve_ivp, seq1d
from PIL import Image
import pdb

# # run on cpu
# jax.config.update('jax_platform_name', 'cpu')
# enable float 64
jax.config.update('jax_enable_x64', True)

class HNNModule(nn.Module):
    nhiddens: int
    nlayers: int
    dtype: Any = jnp.float32

    def setup(self):
        self.layers = [nn.Dense(self.nhiddens, param_dtype=self.dtype, dtype=self.dtype) for _ in range(self.nlayers)]
        self.out = nn.Dense(1, param_dtype=self.dtype, dtype=self.dtype)

    def __call__(self, y: jnp.ndarray) -> jnp.ndarray:
        # compute dy/dt
        # y: (nstates,)
        # output: (1,)
        for layer in self.layers:
            y = layer(y)
            y = jax.nn.softplus(y)
        y = self.out(y)
        return y

class GravitationalDataset:
    # a dataset that returns the time and states of a gravitational system
    def __init__(self, ntpts: int, tmax: float, dset_length: int, key, fpath: str, dtype):
        self.nstates = 8
        self.ntpts = ntpts
        self.tmax = tmax
        self.dset: Dict[int, Tuple[jnp.ndarray, jnp.ndarray]] = {}
        self.dset_length = dset_length
        self.dtype = dtype

        # simulate the dataset
        datapath = os.path.join(fpath, "gravdata.pkl")
        if os.path.exists(datapath):
            with open(datapath, "rb") as f:
                self.tpts, self.ys = pickle.load(f)
                self.tpts = jnp.array(self.tpts, dtype=dtype)  # (ntpts,)
                self.ys = jnp.array(self.ys, dtype=dtype)  # (dset_length, ntpts, nstates)
        else:
            init_states = self._get_init_states(dset_length, key)  # (dset_length, nstates)
            tpts = np.linspace(0, tmax, ntpts)
            all_ys = []
            for i in tqdm(range(dset_length)):
                init_state = init_states[i]  # (nstates,)
                y = scipy.integrate.solve_ivp(
                    self._dydt, (0, tmax), init_state, t_eval=tpts, atol=1e-9, rtol=1e-9).y  # (nstates, ntpts)
                all_ys.append(y.T)  # the element is (ntpts, nstates)
            ys = np.stack(all_ys, axis=0)  # (dset_length, ntpts, nstates)
            self.tpts = jnp.array(tpts, dtype=dtype)  # (ntpts,)
            self.ys = jnp.array(ys, dtype=dtype)  # (dset_length, ntpts, nstates)

            with open(datapath, "wb") as f:
                pickle.dump((tpts, ys), f)

    def getitem(self, idx: int, length: Optional[int] = None) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        y = self.ys[idx]
        idxx = jnp.asarray(idx, dtype=jnp.int32)
        if length is None:
            return self.tpts, y, idxx
        else:
            return self.tpts[:length], y[:length], idxx

    def __len__(self) -> int:
        return self.dset_length

    def _dydt(self, t: float, y: np.ndarray) -> np.ndarray:
        # y: (nstates=8,)
        # returns: (nstates,)
        # state: (x1, y1, x2, y2, vx1, vy1, vx2, vy2)
        r1, r2 = y[..., :2], y[..., 2:4]  # (..., 2)
        d12_32_inv = (np.sum((r1 - r2) ** 2, axis=-1, keepdims=True)) ** (-1.5)
        a1 = d12_32_inv * (r2 - r1)
        a2 = -a1
        dstate = np.concatenate((y[..., 4:], a1, a2), axis=-1)
        return dstate

    def _get_init_states(self, dset_length: int, key: jnp.ndarray) -> np.ndarray:
        # returns: (nstates=8,)
        # state: (x1, y1, x2, y2, vx1, vy1, vx2, vy2)
        key, *subkey = jax.random.split(key, 4)
        r0 = jax.random.uniform(subkey[0], (dset_length,), minval=0.5, maxval=1.5, dtype=self.dtype)
        theta0 = jax.random.uniform(subkey[1], (dset_length,), minval=-np.pi, maxval=np.pi, dtype=self.dtype)
        v0 = jax.random.uniform(subkey[2], (dset_length,), minval=0.7, maxval=1.0, dtype=self.dtype)
        v0 = v0 * (0.5 / r0 ** 0.5)
        x0, y0 = r0 * jnp.cos(theta0), r0 * jnp.sin(theta0)
        state0 = jnp.stack([
            x0, y0, -x0, -y0, -v0 * np.sin(theta0), v0 * np.cos(theta0), v0 * np.sin(theta0), -v0 * np.cos(theta0)],
            axis=-1)  # (dset_length, nstates=8)
        return np.asarray(state0)

def get_hnn_dynamics(model: HNNModule, params: Any, y: jnp.ndarray) -> jnp.ndarray:
    # y: (nstates,)
    # returns: (nstates,)
    jac = jax.jacfwd(model.apply, argnums=1)({"params": params}, y)  # (1, nstates)
    dydt = jnp.concatenate((jac[0, jac.shape[-1] // 2:], -jac[0, :jac.shape[-1] // 2]), axis=-1)  # (nstates,)
    # jax.debug.print("{dydt_shape}, {jac_shape}", dydt_shape=dydt.shape, jac_shape=jac.shape)
    return dydt

@partial(jax.jit, static_argnames=("model", "method"))
def rollout(model: HNNModule, params: Any, y0: jnp.ndarray, tpts: jnp.ndarray, yinit_guess: jnp.ndarray,
            method: str = "deer") -> jnp.ndarray:
    # roll out the model's predictions
    # y0: (nstates,)
    # tpts: (ntpts,)
    # yinit_guess: (ntpts, nstates)
    # returns: (ntpts, nstates)
    model_func = lambda y, x, params: get_hnn_dynamics(model, params, y)
    if method == "deer":
        return solve_ivp(model_func, y0, tpts[..., None], params, tpts, yinit_guess=yinit_guess)
    else:
        return odeint(model_func, y0, tpts, params)

@partial(jax.jit, static_argnames=("model", "method"))
def loss_fn(model: HNNModule, params: Any, batch: Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray],
            all_yinit_guess: jnp.ndarray, weight: jnp.ndarray, method: str="deer") -> jnp.ndarray:
    # compute the loss
    # batch: (batch_size, ntpts), (batch_size, ntpts, nstates), (batch_size)
    # yinit_guess (ndata, ntpts, nstates)
    # weight: (ntpts,)
    tpts, y, idxs = batch
    y0 = y[:, 0, :]  # (batch_size, nstates)

    # get the yinit guess for each batch
    yinit_guess = all_yinit_guess[idxs]  # (batch_size, ntpts, nstates)

    # ypred: (batch_size, ntpts, nstates)
    ypred = jax.vmap(rollout, in_axes=(None, None, 0, 0, 0, None))(model, params, y0, tpts, yinit_guess, method)

    # update the all_yinit_guess
    # jax.debug.print("{ypred}", ypred=ypred)
    all_yinit_guess = all_yinit_guess.at[idxs].set(ypred)  # (ndata, ntpts, nstates)
    # jax.debug.print("{all_yinit_guess} {all_yinit_guess_shape}, {idxs}",
    #                 all_yinit_guess=all_yinit_guess[:10], all_yinit_guess_shape=all_yinit_guess[idxs].shape,
    #                 idxs=idxs)
    dev = (y - ypred) ** 2  # (batch_size, ntpts, nstates)
    # comment/uncomment below to disable/enable progressive weighting over time
    # dev = dev * weight[..., None]  # (batch_size, ntpts, nstates)
    # dev = jnp.sum(dev, axis=-2) / jnp.sum(weight)  # (batch_size, nstates)
    return jnp.mean(dev), all_yinit_guess  # (,), (ndata, ntpts, nstates)

@partial(jax.jit, static_argnames=("model", "optimizer", "method"))
def update_step(model:HNNModule, optimizer: optax.GradientTransformation, params: optax.Params, opt_state: Any,
                batch: Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray], all_yinit_guess: jnp.ndarray,
                weight: jnp.ndarray,
                method: str="deer") \
        -> Tuple[optax.Params, Any, jnp.ndarray, jnp.ndarray]:
    (loss, all_yinit_guess), grad = \
        jax.value_and_grad(loss_fn, argnums=1, has_aux=True)(model, params, batch, all_yinit_guess, weight, method)
    updates, opt_state = optimizer.update(grad, opt_state, params)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss, all_yinit_guess

def main():
    # set up argparse for the hyperparameters above
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--nepochs", type=int, default=999999999)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--version", type=int, default=0)
    parser.add_argument("--method", type=str, default="deer")
    args = parser.parse_args()

    nstates = 8
    dtype = jnp.float64

    # check the path
    logpath = "logs"
    path = os.path.join(logpath, f"version_{args.version}")
    if os.path.exists(path):
        raise ValueError(f"Path {path} already exists!")
    os.makedirs(path, exist_ok=True)

    # set up the model and optimizer
    key = jax.random.PRNGKey(args.seed)
    model = HNNModule(nhiddens=64, nlayers=6, dtype=dtype)
    params = model.init(key, jnp.zeros((nstates,), dtype=dtype))["params"]
    optimizer = optax.adam(learning_rate=args.lr)
    opt_state = optimizer.init(params)

    ntpts = 10000
    tmax = 10
    ntrain, nval, ntest = 800, 100, 100
    ndata = ntrain + nval + ntest
    method = args.method
    subkey, key = jax.random.split(key)
    dset = GravitationalDataset(ntpts, tmax, ndata, subkey, logpath, dtype)
    idxs_train = jnp.arange(ntrain)
    idxs_val = jnp.arange(ntrain, ntrain + nval)
    idxs_test = jnp.arange(ntrain + nval, ntrain + nval + ntest)

    # get the summary writer
    summary_writer = SummaryWriter(log_dir=path)

    # initialize the yinit guess
    all_yinit_guess = jnp.zeros((ndata, ntpts, nstates), dtype=dtype)

    # initialize the weights
    weight = jnp.ones((ntpts,), dtype=dtype)
    iweight = 20
    valweight = jnp.ones((ntpts,), dtype=dtype)

    # training loop
    step = 0
    best_val_loss = 9e99
    for epoch in range(args.nepochs):
        subkey, key = jax.random.split(key)
        idxs_train = jax.random.permutation(subkey, idxs_train)
        summary_writer.add_scalar("epoch", epoch, step)
        # batching the dataset for the training
        for i in tqdm(range(0, ntrain, args.batch_size)):
            batch_idxs = idxs_train[i:i + args.batch_size]
            batch = jax.vmap(dset.getitem, in_axes=(0, None))(batch_idxs, iweight)
            params, opt_state, loss, all_yinit_guess2 = \
                update_step(model, optimizer, params, opt_state, batch, all_yinit_guess[:, :iweight], weight, method)
            all_yinit_guess = all_yinit_guess.at[:, :iweight].set(all_yinit_guess2)
            summary_writer.add_scalar("train_loss", loss, step)
            step += 1

        # update the weight
        iweight = iweight + 20 if iweight < ntpts else ntpts

        # validation, using the same batch size
        val_loss = 0
        for i in range(0, nval, args.batch_size):
            batch_idxs = idxs_val[i:i + args.batch_size]
            batch = jax.vmap(dset.getitem)(batch_idxs)
            new_val_loss, all_yinit_guess = loss_fn(model, params, batch, all_yinit_guess, valweight, method)
            val_loss += new_val_loss * len(batch_idxs)
        val_loss /= nval
        summary_writer.add_scalar("val_loss", val_loss, step)

        if epoch % 5 == 0:
            # generate the trajectories image from the test data
            test_idx = idxs_test[0]
            tpts, yt, test_idx = dset.getitem(test_idx)
            y0 = yt[0]
            yinit_guess = jnp.zeros((ntpts, nstates), dtype=dtype)
            ypred = rollout(model, params, y0, tpts, yinit_guess, method)  # (ntpts, nstates)

            # create temporary image
            plt.figure()
            plt.plot(tpts, yt[..., 0], label="true")
            plt.plot(tpts, ypred[..., 0], label="pred")
            plt.legend()
            plt.savefig(os.path.join(path, f"epoch_{epoch}_test_traj.png"))
            plt.close()

            # load the image as a numpy array
            img = np.array(Image.open(os.path.join(path, f"epoch_{epoch}_test_traj.png")))
            # add the image to the summary writer
            summary_writer.add_image("test_traj", img, step, dataformats="HWC")

        summary_writer.flush()

        # save the model's weights
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            # save the model
            with open(os.path.join(path, "best-model.pt"), "wb") as f:
                # save the parameters and the optimizer's states
                pickle.dump((params, opt_state), f)

if __name__ == "__main__":
    main()
