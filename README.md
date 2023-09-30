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

The files to reproduce the experiments are in the [`experiments/`](experiments/) directory.

The code for reproducing the experiments in the paper is not complete yet and more is coming soon.
We just need time to tidy them up and make them more presentable.

## EigenWorms
To reproduce the experiments with EigenWorms dataset, copy the following command line to go the to directory in which the training code is stored

```
cd deer/_experiments/deer_rnn/
```

### Training
Then, for training, run the following

```
python train.py  --nchannel 1 --precision 32 --batch_size 4 --version 0 --seed 23 --lr 3e-5 --ninps 6 --nstates 32 --nsequence 17984 --nclass 5 --nlayer 5 --dset eigenworms --patience 1000 --patience_metric accuracy
```

The seeds to reproduce the three iterations in the paper are `23, 24, 25`.

If you would like to train the model with sequential RNN to compare the speed as shown in the paper, simply add the flag `--use_scan` in the command line above.

### Inference
For inference, the command line is largely the same as training, except that we run it with `infer.py` and that `batch_size` is changed to 3 so that all `39` test samples are used.

```
python infer.py  --nchannel 1 --precision 32 --batch_size 3 --version 0 --seed 23 --lr 3e-5 --ninps 6 --nstates 32 --nsequence 17984 --nclass 5 --nlayer 5 --dset eigenworms
```

