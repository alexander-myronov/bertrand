import argparse
import logging
import os
import shutil

import matplotlib.pyplot as plt
import pandas as pd

from bertrand.training.utils import get_last_ckpt, load_metrics_df


def parse_args() -> argparse.Namespace:
    """
    argument parser for the script
    :return: args
    """
    parser = argparse.ArgumentParser(
        "This script plots the loss and copies the best checkpoint to output directory"
    )

    parser.add_argument(
        "--input-dir", type=str, required=True, help="Path to model checkpoints",
    )

    parser.add_argument(
        "--out-dir", type=str, required=True, help="Path to save the best model",
    )

    return parser.parse_args()


def plot_loss(metrics: pd.DataFrame, eval_metrics: pd.DataFrame, out_file: str, best_epoch: int, best_steps: int):
    """
    Plots train and eval losses, saves the plot
    :param metrics: full set of metrics
    :param eval_metrics: evaluation set of metrics
    :param out_file: file to save the loss to
    """
    logging.info(f"Plotting loss, saving plot to {out_file}")
    fig, ax = plt.subplots()
    ax.plot(metrics.epoch, metrics.loss, label="train", c="blue")
    ax.plot(eval_metrics.epoch, eval_metrics.eval_loss, label="eval", c="green")
    ax.axvline(best_epoch, linewidth=2, c='r', linestyle='--')
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title(f"Best epoch {best_epoch}, best steps {best_steps}")
    fig.legend()
    fig.savefig(out_file)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    args = parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    ckpt = get_last_ckpt(args.input_dir)
    metrics = load_metrics_df(ckpt)


    # Put evaluation metrics into a separate variable
    eval_metrics = metrics[~metrics.eval_loss.isna()].reset_index(drop=True)

    best_epoch = eval_metrics.eval_loss.idxmin()
    best_steps = eval_metrics.loc[best_epoch, "step"]
    # best_steps = eval_metrics.step.iloc[-1]

    plot_loss(metrics, eval_metrics, os.path.join(args.out_dir, "loss.png"), best_epoch=best_epoch, best_steps=best_steps)

    best_ckpt = os.path.join(args.input_dir, f"checkpoint-{best_steps}")

    logging.info(f"Copying {best_ckpt} to {args.out_dir}")
    shutil.copytree(best_ckpt, args.out_dir, dirs_exist_ok=True)
