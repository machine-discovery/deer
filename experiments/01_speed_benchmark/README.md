# Experiment 01: speed benchmarking

This is to reproduce the Figure 2 in the paper.
There are 3 script files in this directory:

* [`seq1d.py`](seq1d.py): to run the benchmark of forward evaluation only and produce the report
* [`seq1d_val_grad.py`](seq1d_val_grad.py): to run the benchmark of forward evaluation + gradient and produce the report
* [`get_plot_speedup.py`](get_plot_speedup.py): to plot the figures from the report

## Running the experiments

To run the experiments, simply run the script:

```
python -u seq1d.py --batchsize [BATCHSIZE] > [FILE]
python get_plot_speedup.py [FILE] --batchsize [BATCHSIZE]
```

for generating the benchmark for forward evaluation only, or

```
python -u seq1d_val_grad.py --batchsize [BATCHSIZE] > [FILE]
python get_plot_speedup.py [FILE] --batchsize [BATCHSIZE]
```

for generating the benchmark for forward evaluation + gradient.
Note that you need to change `[BATCHSIZE]` to the batch size and `[FILE]` to the file name to save the report to.
In our paper, we use `[BATCHSIZE]` equals to `16`.

**Estimated running time:** hours to 1 day
