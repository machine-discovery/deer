import argparse
import os
import pickle
import torch
import jax
import jax.numpy as jnp
import equinox as eqx
from tqdm import tqdm
from config import read_config, get_model_from_dct, get_case_from_dct
from models import RNNNet  # for pickle
from train import calc_loss


def infer():
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", type=int, required=True)
    parser.add_argument("--logdir", type=str, default="logs")
    args = parser.parse_args()

    key = jax.random.PRNGKey(0)
    key, subkey = jax.random.split(key, 2)

    # get the config file
    FDIR = os.path.abspath(os.path.dirname(os.path.realpath(__file__)))
    logdir = os.path.join(FDIR, args.logdir)
    configfile = os.path.join(logdir, f"config_{args.version}.yaml")
    expdir = os.path.join(logdir, f"version_{args.version}")

    # read the config file
    config = read_config(configfile)
    general_config = config["general"]
    batch_size = general_config["batch_size"]

    # get the case and the model
    case = get_case_from_dct(config["case"])
    model = get_model_from_dct(
        config["model"], num_inps=case.num_inps, num_outs=case.num_outs, with_embedding=case.with_embedding,
        reduce_length=case.reduce_length, key=subkey)
    # split the model into model and static
    _, model_static = eqx.partition(model, eqx.is_inexact_array)

    # load the states from the best model
    best_state_fpath = os.path.join(expdir, "best-states.pkl")
    with open(best_state_fpath, "rb") as fb:
        states = pickle.load(fb)
    model_params = states["model_params"]

    # get the test dataset
    test_dset = torch.utils.data.Subset(case, case.test_idxs)
    test_dloader = torch.utils.data.DataLoader(test_dset, batch_size=batch_size, shuffle=False)

    tot_loss = 0
    tot_count = 0
    for batch_tensor in (tbar := tqdm(test_dloader, leave=False, desc="Test")):
        # get the batch in jax
        batch = tuple([jnp.array(batch_t.numpy()) for batch_t in batch_tensor])
        ndata = len(batch[0])

        # evaluate the model
        loss = calc_loss(model_params, model_static, case.val_loss_fn, batch,
                         key=subkey, inference=True)

        tot_loss = tot_loss + loss * ndata
        tot_count = tot_count + ndata

        tbar.set_description(f"Test: {tot_loss / tot_count:.3e}")

    print(f"Test: {tot_loss / tot_count:.3e}")

if __name__ == "__main__":
    infer()
