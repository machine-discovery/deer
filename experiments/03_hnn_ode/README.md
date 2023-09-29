# Experiment 03: NeuralODE with HNN

This is the experiment on training a NeuralODE with HNN using DEER vs sequential method.
This is to reproduce the Figure 4 in the paper.

To run the experiment, simply execute:

```
python train.py --version 0 --method deer
python train.py --version 1 --method sequential
```

And to reproduce the figure, here are the things you need to do:

1. Run the experiments as above
2. Download the CSV files from Tensorboard for both experiments and put them in this directory
3. Run `python plot_results.py` to generate the figure

**Estimated running time**

The experiment will run as long as you want, but we ran it for at least 1 day for DEER and at least 2 weeks for sequential method to reproduce the figure on the paper.
