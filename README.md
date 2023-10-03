# DEER
The official repository of "Parallelizing non-linear sequential models over the sequence length" paper

## Installation

The experiment we ran uses JAX 0.4.11, which runs fine.
However, when we tried using the latest JAX (0.4.16), it raises an error.
So we highly recommend to install the exact same version as ours for compatibility and reproducibility.
Here are the commands to use the same versions as ours:

```
pip install --upgrade -e .
pip install --upgrade jax==0.4.11 jaxlib==0.4.11+cuda11.cudnn86 -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
pip install nvidia-cudnn-cu11==8.6.0.163
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

A typical output for the demo script using a V100 GPU is as follows (your output may vary):

```
$ python deer/demo.py 
=========================================
Problem setup
-----------------------------------------
* Random seed: 0
* Cell: GRU
* Input size: 2
* Batch size: 16
* Sequence length: 10000
* Data type: float32 with eps = 1.192e-07
=========================================
You can change the problem setup by passing arguments to this script.
To see the list of arguments, run with --help.

Benchmarking sequential method: 0.22577 seconds
Benchmarking DEER: 0.00331 seconds
DEER GRU speed up over sequential GRU: 68.189x
Maximum absolute deviation: 2.384e-07 where output range: -9.216077e-01 to 7.263898e-01
```

The files to reproduce the experiments are in the [`experiments/`](experiments/) directory with reproducibility instructions are mentioned in README.md file in each experiment directory.

Speed comparison of training a GRU model using sequential method (orange) vs DEER method (blue) (2 seconds in this animation corresponds to about an hour in training time):
![rnn_train](https://github.com/mfkasim1/deer/assets/1624640/17659f34-219b-42b9-8683-a858a111dbdc)
