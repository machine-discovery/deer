import math
import matplotlib.pyplot as plt
import numpy as np
import imageio.v2 as imageio
from io import BytesIO
from tqdm import tqdm


def smooth(scalars: list[float], weight: float) -> list[float]:
    """
    EMA implementation according to
    https://github.com/tensorflow/tensorboard/blob/34877f15153e1a2087316b9952c931807a122aa7/tensorboard/components/vz_line_chart2/line-chart.ts#L699
    """
    last = 0
    smoothed = []
    num_acc = 0
    for next_val in scalars:
        last = last * weight + (1 - weight) * next_val
        num_acc += 1
        # de-bias
        debias_weight = 1
        if weight != 1:
            debias_weight = 1 - math.pow(weight, num_acc)
        smoothed_val = last / debias_weight
        smoothed.append(smoothed_val)
    return smoothed


def main():
    smooth_params = 0.6
    fast_result_fname = "run-version_24-tag-val_accuracy.csv"
    slow_result_fname = "run-version_26-tag-val_accuracy.csv"
    fast_label = "DEER method"
    slow_label = "Sequential method"

    # load the data
    fast_result = np.loadtxt(fast_result_fname, delimiter=",", skiprows=1)
    slow_result = np.loadtxt(slow_result_fname, delimiter=",", skiprows=1)

    # get the maximum steps (the 2nd column, index 1)
    fast_max_step = fast_result[:, 1].max()
    slow_max_step = slow_result[:, 1].max()
    max_step = min(fast_max_step, slow_max_step)

    # cut the result table to the maximum steps
    fast_result = fast_result[fast_result[:, 1] <= max_step]
    slow_result = slow_result[slow_result[:, 1] <= max_step]

    # the first column is wall-time, so we need to convert it into relative time
    fast_result[:, 0] = fast_result[:, 0] - fast_result[0, 0]
    slow_result[:, 0] = slow_result[:, 0] - slow_result[0, 0]

    # get the relative time and step
    fast_rel_time = fast_result[:, 0] / 3600
    slow_rel_time = slow_result[:, 0] / 3600
    fast_step = fast_result[:, 1] / 1e3
    slow_step = slow_result[:, 1] / 1e3

    # get the raw and smoothed values
    fast_raw = fast_result[:, 2]
    slow_raw = slow_result[:, 2]
    fast_smooth = smooth(fast_result[:, 2], smooth_params)
    slow_smooth = smooth(slow_result[:, 2], smooth_params)

    # plotting parameters
    alpha = 0.2  # alpha for the row values
    tick_fontsize = 12
    label_fontsize = 16
    legend_fontsize = 14

    images = []
    for idx in tqdm(range(len(slow_rel_time))):
        # plot the results
        fig, ax = plt.subplots(figsize=(15, 4))

        cur_time = slow_rel_time[idx]
        fast_indices = max([i for i, t in enumerate(fast_rel_time) if t <= cur_time])
        if fast_indices:
            ax.plot(fast_step[:fast_indices + 1], fast_raw[:fast_indices + 1], 'C0', alpha=alpha)
            ax.plot(fast_step[:fast_indices + 1], fast_smooth[:fast_indices + 1], 'C0', label=fast_label)
        else:
            ax.plot(fast_step, fast_smooth, 'C0', label=fast_label)

        ax.plot(slow_step[:idx + 1], slow_raw[:idx + 1], 'C1', alpha=alpha)
        ax.plot(slow_step[:idx + 1], slow_smooth[:idx + 1], 'C1', label=slow_label)

        ax.set_xlabel(r"Training steps ($\times 10^3$)" + "\n(d)", fontsize=label_fontsize)
        ax.set_ylabel("Validation accuracy", fontsize=label_fontsize)
        ax.legend(fontsize=legend_fontsize)
        plt.xticks(fontsize=tick_fontsize)
        plt.yticks(fontsize=tick_fontsize)
        plt.tight_layout()

        buf = BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        images.append(imageio.imread(buf))
        plt.close(fig)

    imageio.mimsave("rnn_train.gif", images, duration=2)


if __name__ == "__main__":
    main()
