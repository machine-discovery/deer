
## EigenWorms
To reproduce the experiments with EigenWorms dataset, follow the following instructions in the current directory

### Training
For training, run the following

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

