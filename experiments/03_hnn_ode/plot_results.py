import math
import os
import matplotlib.pyplot as plt
import numpy as np

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
    smooth_params = 0.9
    fast_result_fname = "run-version_0-tag-val_loss.csv"
    slow_result_fname = "run-version_1-tag-val_loss.csv"
    fast_label = "DEER method"
    slow_label = "RK45 method"

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

    # plot the results
    plt.figure(figsize=(15, 4))
    plt.subplot(1, 2, 1)
    plt.plot(fast_rel_time, fast_raw, 'C0', alpha=alpha)
    plt.plot(slow_rel_time, slow_raw, 'C1', alpha=alpha)
    plt.plot(fast_rel_time, fast_smooth, 'C0', label=fast_label)
    plt.plot(slow_rel_time, slow_smooth, 'C1', label=slow_label)
    plt.gca().set_yscale("log")
    # plot an arrow from the slow to fast's last smoothed value
    plt.annotate(
        "",
        xy=(fast_rel_time[-1], fast_smooth[-1]),
        xytext=(slow_rel_time[-1], fast_smooth[-1]),
        arrowprops=dict(arrowstyle="->", color="C2", linewidth=2),
    )
    # add a text "nx faster-" underneath the middle part of the arrow
    arrow_x = (slow_rel_time[-1] + fast_rel_time[-1]) / 2
    arrow_y = fast_smooth[-1] / 1.3
    naccel = slow_rel_time[-1] / fast_rel_time[-1]
    plt.text(arrow_x, arrow_y, "%dx faster" % np.round(naccel), va="top", ha="center", backgroundcolor="w", fontsize=12)
    plt.xticks(fontsize=tick_fontsize)
    plt.yticks(fontsize=tick_fontsize)
    plt.xlabel("Hours\n(a)", fontsize=label_fontsize)
    plt.ylabel("Validation loss", fontsize=label_fontsize)

    plt.subplot(1, 2, 2)
    plt.plot(fast_step, fast_raw, 'C0', alpha=alpha)
    plt.plot(slow_step, slow_raw, 'C1', alpha=alpha)
    plt.plot(fast_step, fast_smooth, 'C0', label=fast_label)
    plt.plot(slow_step, slow_smooth, 'C1', label=slow_label)
    plt.gca().set_yscale("log")
    plt.xlabel(r"Training steps ($\times 10^3$)" + "\n(b)", fontsize=label_fontsize)
    plt.legend(fontsize=legend_fontsize)
    plt.xticks(fontsize=tick_fontsize)
    plt.yticks(fontsize=tick_fontsize)

    plt.tight_layout()
    plt.savefig("ode_hnn_train_comparison.png")
    plt.close()

if __name__ == "__main__":
    main()
