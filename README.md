# DEER
The official repository of "Parallelizing non-linear sequential models over the sequence length" paper

## Installation

The experiment we ran uses JAX 0.4.11, which runs fine.
However, when we tried using the latest JAX (0.4.16), it raises an error.
So we highly recommend to install the exact same version as ours for compatibility and reproducibility.
Here are the commands to use the same versions as ours:

```
pip install --upgrade jax==0.4.11 jaxlib==0.4.11+cuda12.cudnn88 -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
pip install --upgrade -e .
```

## Getting started

The best way to get started is to run the demo script by

```
python deer/demo.py
```

This will run a simple speed comparison between DEER and sequential method.
The demo script has various options which can be seen by `python deer/demo.py --help`.

## File guide

On the `deer/` directory:

* [`deer_iter.py`](deer/deer_iter.py): the implementation of the DEER iterations, including forward evaluation and backward gradients.
* [`seq1d.py`](deer/seq1d.py): materialization of DEER for discrete 1D sequences and NeuralODE.
* [`demo.py`](deer/demo.py): a demo script to run the discrete 1D sequence experiment using untrained GRU from Equinox.

The files to reproduce the experiments are in the [`experiments/`](experiments/) directory with reproducibility instructions are mentioned in README.md file in each experiment directory.
