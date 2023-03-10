import argparse
import logging
import os
from typing import Tuple

import pandas as pd


def parse_args() -> argparse.Namespace:
    """
    argument parser for the script
    :return: args
    """
    parser = argparse.ArgumentParser(
        "This script generates a hypothetical peptide:TCR repertoire\n"
        "by randomly pairing MHC-I-presented peptides from mass spectrometry experiments\n"
        "with synthetic TCR generated by `bertrand/utils/immunesim.R`\n"
        "The hypothetical peptide:TCR repertoire is then split into\n"
        "train and validation sets"
    )
    parser.add_argument(
        "--input-peptides", type=str, required=True, help="Path to peptide dataset",
    )
    parser.add_argument(
        "--input-tcrs", type=str, required=True, help="Path to TCR dataset",
    )

    parser.add_argument(
        "--out-dir",
        type=str,
        required=True,
        help="Path to save train and validation sets.\n",
    )

    parser.add_argument(
        "--pair-seed", type=int, default=42, help="Seed for random pairing",
    )

    parser.add_argument(
        "--split-seed", type=int, default=42, help="Seed for train/val split",
    )

    parser.add_argument(
        "--split-frac",
        type=float,
        default=0.2,
        help="Fraction of TCRs in the validation set",
    )

    return parser.parse_args()


def read_tcrs(fn: str) -> pd.DataFrame:
    """
    Read synthetic TCRs generated using `immunesim.R`
    :param fn: filename
    :return: filtered dataset of unique synthetic TCRs
    """
    logging.info("Reading TCRs")
    synthetic_tcrs = pd.read_csv(fn, index_col=0)
    logging.info(f"{len(synthetic_tcrs)} TCRs read")
    synthetic_tcrs = synthetic_tcrs.junction_aa.drop_duplicates()
    synthetic_tcrs = synthetic_tcrs[
        (synthetic_tcrs.str.len() >= 10)
        & (synthetic_tcrs.str.len() <= 20)
        & (synthetic_tcrs.str[0] == "C")
    ]
    logging.info(f"{len(synthetic_tcrs)} unique TCRs after filtering")
    return synthetic_tcrs


def read_peptides(fn: str) -> pd.DataFrame:
    """
    Read presented peptides
    :param fn: filename
    :return: dataset of unique presented peptides
    """
    logging.info("Reading peptides")
    presented_peptides = pd.read_csv(fn, index_col=0)
    logging.info(f"{len(presented_peptides)} peptides read")
    presented_unique = (
        presented_peptides.reset_index()
        .groupby("Peptide2")
        .agg(
            {
                "HLA_type": lambda x: "|".join(sorted(x)),
                "index": lambda x: "|".join(sorted(x)),
            }
        )
        .reset_index()
    )
    logging.info(f"{len(presented_unique)} unique peptides")
    return presented_unique


def sample_peptide_tcr_repertoire(
    presented_peptides: pd.DataFrame, synthetic_tcrs: pd.DataFrame, seed: int
):
    """
    Randomly pair peptides and TCRs
    As there are less peptide than than TCRs,
    peptides are sampled with replacement.
    :param presented_peptides: dataset of presented peptides
    :param synthetic_tcrs: dataset of synthetic TCRs
    :param seed: random state for sampling
    :return: hypothetical peptide:TCR repertoire
    """
    logging.info(
        f"Sampling 1 of {len(presented_peptides)} peptides "
        f"for {len(synthetic_tcrs)} TCRs with replacement"
    )
    peptides_sampled = presented_peptides.sample(
        len(synthetic_tcrs), random_state=seed, replace=True
    )

    peptides_sampled.loc[:, "CDR3b"] = synthetic_tcrs.values
    peptide_tcr_repertoire = peptides_sampled.rename(columns={"Peptide2": "Peptide"})
    return peptide_tcr_repertoire


def split_train_val(
    peptide_tcr_repertoire: pd.DataFrame, seed: int, frac: float
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split peptide:TCR repertoire intro train and validation subsets.
    `frac` of TCRs are sampled for the validation set.
    Peptide:TCR pairs with TCRs from the validation set
    are removed from the training set.
    :param peptide_tcr_repertoire: peptide:TCR repertoire
    :param frac: fraction of TCRs for the validation set
    :param seed: random state for TCR sampling
    :return: training and validation sets
    """
    val_cdr3b = peptide_tcr_repertoire.CDR3b.drop_duplicates().sample(
        frac=frac, replace=False, random_state=seed
    )
    val_set = peptide_tcr_repertoire[peptide_tcr_repertoire.CDR3b.isin(val_cdr3b)]
    train_set = peptide_tcr_repertoire[~peptide_tcr_repertoire.CDR3b.isin(val_cdr3b)]
    logging.info(f"Train: {len(train_set)}, val: {len(val_set)}")
    return train_set, val_set


def save_files(out_dir: str, train: pd.DataFrame, val: pd.DataFrame) -> None:
    """
    Saves the files for MLM pre-training
    :param out_dir: output directory
    :param train: training set
    :param val: validation set
    """
    logging.info(f"Saving `mlm_train.csv.gz` and `mlm_val.csv.gz` to {out_dir}")
    train[["HLA_type", "Peptide", "CDR3b"]].to_csv(
        os.path.join(out_dir, "mlm_train.csv.gz")
    )
    val[["HLA_type", "Peptide", "CDR3b"]].to_csv(
        os.path.join(out_dir, "mlm_val.csv.gz")
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    args = parse_args()

    presented_peptides = read_peptides(args.input_peptides)
    synthetic_tcrs = read_tcrs(args.input_tcrs)
    peptide_tcr_repertoire = sample_peptide_tcr_repertoire(
        presented_peptides, synthetic_tcrs, args.pair_seed
    )
    train, val = split_train_val(
        peptide_tcr_repertoire, args.split_seed, args.split_frac
    )
    save_files(args.out_dir, train, val)
