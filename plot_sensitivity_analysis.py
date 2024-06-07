import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pickle
import os
from typing import Optional
import pandas as pd
import seaborn as sns
import sys


def statistics_table(data, variable: str, save_path: Optional[str] = None):
    steps = list(sorted([run[variable] for run in data["run_results"]]))
    repeats = list(np.arange(data["n_repeats"]))

    steps_idx = dict(zip(steps, range(len(steps))))

    cum_prob_matrix = np.zeros((len(repeats), len(steps)))
    point01_matrix = np.zeros((len(repeats), len(steps)))
    point05_matrix = np.zeros((len(repeats), len(steps)))
    mean_dist_matrix = np.zeros((len(repeats), len(steps)))
    median_dist_matrix = np.zeros((len(repeats), len(steps)))
    p1_matrix = np.zeros((len(repeats), len(steps)))
    timing_matrix = np.zeros((len(repeats), len(steps)))

    for step in data["run_results"]:
        for run in step["runs"]:
            cum_prob_matrix[run["run"], steps_idx[step[variable]]] = run["cum_probability"]
            point01_matrix[run["run"], steps_idx[step[variable]]] = run["point01"]
            point05_matrix[run["run"], steps_idx[step[variable]]] = run["point05"]
            mean_dist_matrix[run["run"], steps_idx[step[variable]]] = run["mean_distance"]
            median_dist_matrix[run["run"], steps_idx[step[variable]]] = run["median_distance"]
            p1_matrix[run["run"], steps_idx[step[variable]]] = run["probabilities"][0]
            timing_matrix[run["run"], steps_idx[step[variable]]] = run["timing"][-1][1]

    avg_p1 = np.mean(p1_matrix, axis=0)
    std_p1 = np.std(p1_matrix, axis=0)
    avg_cp = np.mean(cum_prob_matrix, axis=0)
    std_cp = np.std(cum_prob_matrix, axis=0)
    avg_psi01 = np.mean(point01_matrix, axis=0)
    std_psi01 = np.std(point01_matrix, axis=0)
    avg_psi05 = np.mean(point05_matrix, axis=0)
    std_psi05 = np.std(point05_matrix, axis=0)
    avg_mean_dist = np.mean(mean_dist_matrix, axis=0)
    std_mean_dist = np.std(mean_dist_matrix, axis=0)
    avg_median_dist = np.mean(median_dist_matrix, axis=0)
    std_median_dist = np.std(median_dist_matrix, axis=0)
    avg_runtime = np.mean(timing_matrix, axis=0)
    std_runtime = np.mean(timing_matrix, axis=0)

    print(f"avg_p1: {avg_p1}")
    print(f"std_p1: {std_p1}")
    print(f"avg_cp: {avg_cp}")
    print(f"std_cp: {std_cp}")
    print(f"avg_psi01: {avg_psi01}")
    print(f"std_psi01: {std_psi01}")
    print(f"avg_psi05: {avg_psi05}")
    print(f"std_psi05: {std_psi05}")
    print(f"avg_mean_dist: {avg_mean_dist}")
    print(f"std_mean_dist: {std_mean_dist}")
    print(f"avg_median_dist: {avg_median_dist}")
    print(f"std_mean_dist: {std_median_dist}")
    print(f"avg_runtime: {avg_runtime}")
    print(f"std_runtime: {std_runtime}")

    statistics_df = pd.DataFrame({
        "avg_p1": avg_p1,
        "avg_cp": avg_cp,
        "avg_psi01": avg_psi01,
        "avg_mean_dist": avg_mean_dist,
        "avg_psi05": avg_psi05,
        "avg_median_dist": avg_median_dist,
        "avg_runtime": avg_runtime,
        "std_p1": std_p1,
        "std_cp": std_cp,
        "std_psi01": std_psi01,
        "std_psi05": std_psi05,
        "std_mean_dist": std_mean_dist,
        "std_median_dist": std_median_dist,
        "std_runtime": std_runtime,
    }, index=steps)

    if save_path is not None:
        statistics_df.transpose().to_excel(os.path.join(save_path, "statistics.xlsx"))

    return (p1_matrix, cum_prob_matrix, point05_matrix, mean_dist_matrix), statistics_df, steps


def plot_over_var(statistics_df, x_var, y_var, logscale=True, save_path=None, xlabel=None, ylabel=None):
    ax = sns.lineplot(statistics_df.reset_index(names=[x_var]), x=x_var, y=y_var)
    if xlabel is not None:
        plt.xlabel(xlabel)
    if ylabel is not None:
        plt.ylabel(ylabel)

    if logscale and y_var not in ["avg_runtime", "avg_mean_dist"]:
        plt.yscale("log")
        ax.yaxis.set_major_locator(mticker.LogLocator())

    if y_var == "avg_runtime" and x_var == "K":
        ax.yaxis.set_major_locator(mticker.MultipleLocator(0.1))
        ax.yaxis.set_minor_locator(mticker.MultipleLocator(0.05))

    if x_var == "n_acts":
        ax.set_xticks(statistics_df.index)
        if y_var in ["avg_p1", "avg_cp"]:
            ax.yaxis.set_major_locator(mticker.LogLocator(base=10, numticks=8))

    if x_var == "K":
        ax.xaxis.set_major_locator(mticker.MultipleLocator(5000))
        ax.xaxis.set_minor_locator(mticker.MultipleLocator(1000))

    if x_var == "n_events":
        ax.xaxis.set_major_locator(mticker.MultipleLocator(500))
        ax.xaxis.set_minor_locator(mticker.MultipleLocator(100))
        if y_var == "avg_runtime":
            bottom, top = plt.ylim()
            plt.ylim(bottom, 2)
            ax.yaxis.set_minor_locator(mticker.MultipleLocator(0.2))
        if y_var in ["avg_p1", "avg_cp"]:
            ax.yaxis.set_major_locator(mticker.LogLocator(base=10, numticks=8))

    # Fix the cases where autoscaling the y-axis doesn't work properly
    if x_var == "K" and y_var == "avg_p1":
        bottom, top = plt.ylim()
        plt.ylim((bottom / 50, top * 50))
        ax.yaxis.set_major_locator(mticker.LogLocator(10, numticks=5))
        locmin = mticker.LogLocator(base=10.0, subs=(0.2, 0.4, 0.6, 0.8), numticks=12)
        ax.yaxis.set_minor_locator(locmin)

    if x_var == "beta":
        if y_var == "avg_runtime":
            bottom, top = plt.ylim()
            distance = top - bottom
            plt.ylim((0, 0.5))
            ax.yaxis.set_major_locator(mticker.MultipleLocator(0.2))
            ax.yaxis.set_minor_locator(mticker.MultipleLocator(0.1))
        if y_var in ["avg_p1", "avg_cp"]:
            ax.yaxis.set_major_locator(mticker.LogLocator(base=10, numticks=8))

    if x_var == "n_acts" and y_var == "avg_runtime":
        bottom, top = plt.ylim()
        plt.ylim((0, 0.8))
        ax.yaxis.set_major_locator(mticker.MultipleLocator(0.25))
        ax.yaxis.set_minor_locator(mticker.MultipleLocator(0.125))

    if x_var == "n_acts" and y_var in ["avg_p1", "avg_cp"]:
        bottom, top = plt.ylim()
        plt.ylim((bottom / 2, top * 2))

    if y_var == "avg_mean_dist":
        bottom, top = plt.ylim()
        plt.ylim((0, top + 1))

    if y_var == "avg_mean_dist" and x_var == "beta":
        #ax.yaxis.set_minor_formatter(mticker.ScalarFormatter())
        ax.yaxis.set_major_locator(mticker.MultipleLocator(5))
        ax.yaxis.set_minor_locator(mticker.MultipleLocator(1))

    if y_var == "avg_mean_dist" and x_var in ["n_events", "n_acts", "K"]:
        ax.yaxis.set_major_locator(mticker.MultipleLocator(1))
        ax.yaxis.set_minor_locator(mticker.MultipleLocator(0.5))

    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, bbox_inches="tight", dpi=300)

    plt.show()


def plot_results(RUN_DIRS):
    xlabels = {
        "n_events": r"$n_{\mathit{events}}$",
        "n_acts": r"$n_{\mathit{act}}$",
        "beta": r"$\beta$",
        "K": r"$K$",
    }

    ylabels = {
        "avg_p1": r"$P(L_1)$",
        "avg_cp": r"$F_K(K)$",
        "avg_psi05": r"$\psi_{0.5}$",
        "avg_mean_dist": r"$d_{\mathit{avg}}$",
        "avg_runtime": r"$t$ in seconds",
    }
    y_save_names = {
        "avg_p1": "p1",
        "avg_cp": "cp",
        "avg_psi05": "psi05",
        "avg_mean_dist": "d_avg",
        "avg_runtime": "runtime",
    }

    sns.set(rc={'text.usetex' : True, "xtick.bottom": True, "ytick.left": True, 'figure.figsize': (6, 4)})
    sns.set_context("poster")
    sns.set_style("ticks")

    for x_var in RUN_DIRS.keys():
        save_dir = os.path.join("evaluation", "results", "plots", x_var)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        y_vars = ["avg_p1", "avg_cp", "avg_runtime", "avg_mean_dist"]

        RUN_DIR = RUN_DIRS[x_var]

        res_dir = RUN_DIR
        filenames = next(os.walk(res_dir), (None, None, []))[2]

        with open(os.path.join(res_dir, "results.pkl"), "rb") as file:
            data = pickle.load(file)

        #statistics_table(data, "n_events", os.path.join(DATA_DIR, RUN_DIR))
        (p1_matrix, cum_prob_matrix, point05_matrix, mean_dist_matrix), statistics_df, steps = statistics_table(data, x_var)

        for y_var in y_vars:
            save_path = os.path.join(save_dir, f"{y_save_names[y_var]}.png")
            plot_over_var(statistics_df, x_var=x_var, y_var=y_var, xlabel=xlabels[x_var], ylabel=ylabels[y_var], save_path=save_path)


if __name__ == "__main__":
    filename = os.path.join("evaluation", "results", "sensitivity_analysis", sys.argv[1])
    with open(filename, "rb") as file:
        RUN_DIRS = pickle.load(file)
    plot_results(RUN_DIRS)
