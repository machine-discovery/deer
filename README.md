# DEER: parallelizing sequential models

Repository for "Parallelizing non-linear sequential models over the sequence length" paper.

To set up the conda environment and install the required packages, from the directory in which `qpert` is stored, copy the following command lines

```
cd qpert/
conda create -n deer -y
conda activate deer
conda install pip -y
pip install -e .
```

To reproduce the experiment with EigenWorms dataset, from the directory `qpert`, copy the following command line to go the to directory in which the training code is stpred

```
cd qpert/deer/experiments/deer_rnn/
```

## Training
Then, for training, run the following

```
python train.py  --nchannel 1 --precision 32 --batch_size 4 --version 0 --seed 23 --lr 3e-5 --ninps 6 --nstates 32 --nsequence 17984 --nclass 5 --nlayer 5 --dset eigenworms --patience 1000
```

The seeds to reproduce the three iterations in the paper are `23, 24, 25`.

## Inference
For inference, the command line is largely the same as training, except that we run it with `infer.py` and that `batch_size` is changed to 3 so that all `39` test samples are used.

```
python infer.py  --nchannel 1 --precision 32 --batch_size 3 --version 0 --seed 23 --lr 3e-5 --ninps 6 --nstates 32 --nsequence 17984 --nclass 5 --nlayer 5 --dset eigenworms
```
