# Experiment 04: DEER and Sequential RNN on EigenWorms Dataset

To reproduce the experiments with EigenWorms dataset as reported in Table 1 in the paper, follow the following instructions in the current directory

### Training
For training with GRU, run the following

```
python train.py --nchannel 1 --precision 32 --batch_size 4 --version 0 --seed 0 --lr 3e-5 --ninps 6 --nstates 32 --nsequence 17984 --nclass 5 --nlayer 5 --dset eigenworms --patience 1000 --patience_metric accuracy
```

Similarly, for training with LEM, run the following

```
python train_lem.py --precision 32 --batch_size 3 --version 0 --seed 0 --lr 3e-2 --ninps 6 --nstates 32 --nsequence 17984 --nclass 5 --dset eigenworms --nepochs 14
```

The seeds to reproduce the three iterations in the paper are `0, 1, 2`. To replicate the results of the paper, make sure that memory preallocation is set to true by using the command line below

```
export XLA_PYTHON_CLIENT_PREALLOCATE=true
```

If you would like to train the model with sequential RNN to compare the speed as shown in the paper, simply add the flag `--use_scan` in the command line above.

### Inference
For inference, the command line is largely the same as training, except that we run it with `infer.py` and that `batch_size` is changed to 3 so that all `39` test samples are used.

```
python infer.py --nchannel 1 --precision 32 --batch_size 3 --version 0 --seed 0 --ninps 6 --nstates 32 --nsequence 17984 --nclass 5 --nlayer 5 --dset eigenworms
```

Similarly, for inference with LEM

```
python infer_lem.py --precision 32 --batch_size 3 --version 0 --seed 0 --ninps 6 --nstates 32 --nsequence 17984 --nclass 5 --dset eigenworms
```

**Estimated running time**

The experiment will run as long as you want, but we ran it for at least 2 hours with DEER and at least 2 days to get the equivalent results with the sequential method. With the patience for early stopping set at 1000 epochs, the training converged after around 4.5 hours for DEER and 4 days and 20 hours for sequential method to reproduce the results in the paper.
