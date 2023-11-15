from typing import Tuple, Callable
import os
import shutil
import pickle
import argparse
from functools import partial
from tqdm import tqdm
import torch
import jax
import jax.numpy as jnp
import equinox as eqx
from tensorboardX import SummaryWriter
import optax
from config import read_config, get_model_from_dct, get_case_from_dct, get_optimizer_from_dct


@partial(jax.jit, static_argnames=("model_static", "loss_fn", "inference"))
def calc_loss(model_params, model_static, loss_fn: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray],
              batch: Tuple[jnp.ndarray, jnp.ndarray],
              key: jax.random.PRNGKeyArray, inference: bool
              ):
    model = eqx.combine(model_params, model_static)
    # input: (batch_size, length, input_size) or (length, batch_size) int
    # target: (batch_size, length, output_size) or (length, batch_size) int
    input, target = batch
    # output: (batch_size, length, output_size)
    output = jax.vmap(model, in_axes=(0, None, None))(input, key, inference)
    # output = jax.vmap(model)(input)
    loss = jnp.mean(jax.vmap(loss_fn)(output, target))
    return loss

@partial(jax.jit, static_argnames=("model_static", "loss_fn", "optimizer"))
def update_step(model_params, model_static, loss_fn: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray],
                optimizer: optax.GradientTransformation, opt_state, batch: Tuple[jnp.ndarray, jnp.ndarray],
                key: jax.random.PRNGKeyArray,
                ):
    loss, grads = jax.value_and_grad(calc_loss)(
        model_params, model_static, loss_fn, batch,
        key=key, inference=False,
    )
    updates, opt_state = optimizer.update(grads, opt_state, model_params)
    model_params = optax.apply_updates(model_params, updates)
    return model_params, opt_state, loss

def train():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--version", type=int, required=True)
    parser.add_argument("--logdir", type=str, default="logs")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--load", type=int, default=None)
    args = parser.parse_args()

    # set up the experimental directory
    FDIR = os.path.abspath(os.path.dirname(os.path.realpath(__file__)))
    logdir = os.path.join(FDIR, args.logdir)
    os.makedirs(logdir, exist_ok=True)
    shutil.copyfile(args.config, os.path.join(logdir, f"config_{args.version}.yaml"))
    path = os.path.join(logdir, f"version_{args.version}")
    if os.path.exists(path):
        raise RuntimeError(f"The log directory {path} already exists")
    best_state_fpath = os.path.join(path, "best-states.pkl")

    # set the random seed
    key = jax.random.PRNGKey(args.seed)
    key, subkey = jax.random.split(key, 2)

    # read the config file
    config = read_config(args.config)
    general_config = config["general"]
    batch_size = general_config["batch_size"]
    max_steps = general_config["max_steps"]

    # get the case and the model
    case = get_case_from_dct(config["case"])
    model = get_model_from_dct(
        config["model"], num_inps=case.num_inps, num_outs=case.num_outs, with_embedding=case.with_embedding,
        reduce_length=case.reduce_length, key=subkey)
    optimizer = get_optimizer_from_dct(config["optimizer"])
    print(optimizer)

    # split the model into model and static
    model_params, model_static = eqx.partition(model, eqx.is_inexact_array)
    opt_state = optimizer.init(model_params)
    # print(opt_state)

    # print the number of parameters in the model
    params_lst = jax.tree_util.tree_flatten(model_params)[0]
    num_params = sum([p.size for p in params_lst if p is not None])
    print(f"Number of model's parameters: {num_params}")

    # get the summary writer
    summary_writer = SummaryWriter(log_dir=path)

    # get the validation dataset
    val_dset = torch.utils.data.Subset(case, case.val_idxs)
    val_dloader = torch.utils.data.DataLoader(val_dset, batch_size=batch_size, shuffle=False)

    isteps = 0
    iepoch = 0
    best_val_loss = float("inf")

    # load the experiment state
    if args.load is not None:
        with open(os.path.join(logdir, f"version_{args.load}", "best-states.pkl"), "rb") as f:
            states = pickle.load(f)
        isteps = states["isteps"]
        iepoch = states["iepoch"]
        best_val_loss = states["best_val_loss"]
        # best_val_loss = float('inf')
        key = states["key"]
        model_params = states["model_params"]
        opt_state = states["opt_state"]

    while isteps < max_steps:
        key, *subkey = jax.random.split(key, 3)

        # start a new epoch by shuffling the train idxs
        train_idxs = jax.random.permutation(subkey[0], case.train_idxs)  # (num_train,)
        train_dset = torch.utils.data.Subset(case, train_idxs)
        train_dloader = torch.utils.data.DataLoader(train_dset, batch_size=batch_size, shuffle=False)
        for batch_tensor in (tbar := tqdm(train_dloader, leave=False)):
            key, subkey = jax.random.split(key, 2)
            # get the batch in jax
            batch = tuple([jnp.array(batch_t.numpy()) for batch_t in batch_tensor])

            # update the model
            model_params, opt_state, loss = update_step(model_params, model_static, case.train_loss_fn,
                                                        optimizer, opt_state, batch,
                                                        key=subkey,
                                                        )
            isteps += 1

            # log the loss
            summary_writer.add_scalar("loss/train", loss, isteps)
            tbar.set_description(f"Epoch {iepoch} loss: {loss:.3e}")

        # evaluate the validation dataset
        tot_loss = 0
        tot_count = 0
        for batch_tensor in (vbar := tqdm(val_dloader, leave=False)):
            key, subkey = jax.random.split(key, 2)
            # get the batch in jax
            batch = tuple([jnp.array(batch_t.numpy()) for batch_t in batch_tensor])
            ndata = len(batch[0])

            # evaluate the model
            loss = calc_loss(model_params, model_static, case.val_loss_fn, batch,
                             key=subkey, inference=True,
                             )
            tot_loss = tot_loss + loss * ndata
            tot_count = tot_count + ndata

            vbar.set_description(f"Epoch {iepoch} val loss: {tot_loss / tot_count:.3e}")

        # log the loss
        val_loss = tot_loss / tot_count
        summary_writer.add_scalar("loss/val", val_loss, isteps)
        iepoch += 1

        # save the best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            states = {
                "isteps": isteps,
                "iepoch": iepoch,
                "best_val_loss": val_loss,
                "key": key,
                "model_params": model_params,
                "opt_state": opt_state,
            }
            with open(best_state_fpath, "wb") as f:
                pickle.dump(states, f)

        # log the epoch vs steps
        summary_writer.add_scalar("epoch", iepoch, isteps)

if __name__ == "__main__":
    train()
