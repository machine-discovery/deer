import os
from typing import Any, Tuple
from functools import partial
import argparse
import jax
import jax.numpy as jnp
import optax
import numpy as np
from tensorboardX import SummaryWriter
from flax import linen as nn
from tqdm import tqdm
from deer.experiments.ultra_deep_mfn.data import ImageDataset
from deer.experiments.ultra_deep_mfn.models import MFNSine, MFNSineLong, SIRENLong, MFNGaborLong


jax.config.update("jax_enable_x64", True)
FDIR = os.path.dirname(os.path.realpath(__file__))

@partial(jax.jit, static_argnames=("model",))
def loss_fn(model: nn.Module, params: Any, batch: Tuple[jnp.ndarray, jnp.ndarray]) -> jnp.ndarray:
    # batch: (coords, values): (nbatch, ncoords), (nbatch, nchannels)
    coords, values = batch
    preds = jax.vmap(model.apply, in_axes=(None, 0))({"params": params}, coords)
    return jnp.mean(jnp.square(preds - values))

@partial(jax.jit, static_argnames=("model", "optimizer"))
def update_step(model: nn.Module, optimizer: optax.GradientTransformation, params: optax.Params, opt_state: Any,
                batch: Tuple[jnp.ndarray, jnp.ndarray]) -> Tuple[optax.Params, Any, jnp.ndarray]:
    loss, grad = jax.value_and_grad(loss_fn, argnums=1, has_aux=False)(model, params, batch)
    updates, opt_state = optimizer.update(grad, opt_state, params)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss

def train():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="mfnsine")
    parser.add_argument("--dataset", type=str, default="image")
    parser.add_argument("--dtype", type=str, default="float32")
    parser.add_argument("--version", type=int, default=0)
    parser.add_argument("--nepochs", type=int, default=10000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=1000000)
    args = parser.parse_args()

    # create the experiment path
    logpath = "logs"
    path = os.path.join(logpath, f"version_{args.version}")
    if os.path.exists(path):
        raise ValueError(f"Path {path} already exists!")
    os.makedirs(path, exist_ok=True)

    # set the dtype
    dtype = {
        "float32": jnp.float32,
        "float64": jnp.float64,
    }[args.dtype]

    # get the dataset
    key = jax.random.PRNGKey(0)
    if args.dataset == "image":
        key, *subkey = jax.random.split(key, 2)
        dset = ImageDataset(
            image_path=os.path.join(FDIR, "media", "henry-be-IicyiaPYGGI-unsplash.jpg"),
            maxnpts=args.batch_size,
            rngkey=subkey[0],
            dtype=dtype,
        )
    else:
        raise NotImplementedError(f"Unknown dataset {args.dataset}")

    # construct the model
    if args.model == "mfnsine":
        model = MFNSine(
            ninputs=dset.ncoords,
            noutputs=dset.nchannels,
            nhiddens=256,
            nlayers=3,
            input_scale=np.max(dset.meshshape),
            dtype=dtype,
        )
    elif args.model == "mfnsine2":
        model = MFNSine(
            ninputs=dset.ncoords,
            noutputs=dset.nchannels,
            nhiddens=2,
            nlayers=1000,
            input_scale=np.max(dset.meshshape),
            dtype=dtype,
        )
    elif args.model == "mfnsinelong":
        model = MFNSineLong(
            ninputs=dset.ncoords,
            noutputs=dset.nchannels,
            nhiddens=8,
            nlayers=1000,
            input_scale=np.max(dset.meshshape),
            dtype=dtype,
        )
    elif args.model == "mfngaborlong":
        model = MFNGaborLong(
            ninputs=dset.ncoords,
            noutputs=dset.nchannels,
            nhiddens=4,
            nlayers=1000,
            input_scale=np.max(dset.meshshape),
            dtype=dtype,
        )
    elif args.model == "sirenlong":
        model = SIRENLong(
            ninputs=dset.ncoords,
            noutputs=dset.nchannels,
            nhiddens=2,
            nlayers=1000,
            w0=10.0,
            dtype=dtype,
        )
    else:
        raise NotImplementedError(f"Unknown model {args.model}")

    # initialize the model's and optimizer's parameters
    key, *subkey = jax.random.split(key, 2)
    params = model.init(subkey[0], jnp.zeros((dset.ncoords,), dtype=dtype))["params"]
    optimizer = optax.adam(learning_rate=args.lr)
    opt_state = optimizer.init(params)

    # training loop
    ntrain = len(dset)
    step = 0
    best_val_loss = 9e99
    summary_writer = SummaryWriter(log_dir=path)
    model_apply_jit = jax.jit(model.apply)
    for epoch in (pbar := tqdm(range(args.nepochs))):
        summary_writer.add_scalar("epoch", epoch, step)
        batch = dset.__getitem__(0)
        params, opt_state, loss = update_step(model, optimizer, params, opt_state, batch)
        summary_writer.add_scalar("train_loss", loss, step)
        step += 1
        pbar.set_description(f"{loss:.3e}")

        if (epoch + 1) % 100 == 0:
            # validation: get the images of the model's output
            coords = dset.getcoords()  # (*meshshape, ncoords)

            # reshape the coordinates
            meshshape = coords.shape[:-1]
            coords = coords.reshape((-1, coords.shape[-1]))  # (nbatch, ncoords)

            # evaluate the values at each coordinate
            model_outputs = []
            for idx in tqdm(range(0, coords.shape[0], args.batch_size), leave=False):
                coords0 = coords[idx:idx + args.batch_size]
                model_output = jax.vmap(model_apply_jit, in_axes=(None, 0))({"params": params}, coords0)
                model_outputs.append(model_output)

            # concatenate all the outputs and restore the meshshape
            model_output = jnp.concatenate(model_outputs, axis=0)
            model_output = model_output.reshape((*meshshape, dset.nchannels))
            img = dset.denorm(model_output)

            summary_writer.add_image("output", np.moveaxis(np.array(img), -1, 0), step)

        if (epoch + 1) % 10 == 0:
            summary_writer.flush()

if __name__ == "__main__":
    train()
