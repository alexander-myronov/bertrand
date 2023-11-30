import argparse
import logging
import os
import shutil

import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from transformers import TrainingArguments, DataCollatorWithPadding, Trainer

from bertrand.immrep.training.cv import train_test_generator
from bertrand.immrep.training.dataset_tcr import TCRDataset
from bertrand.immrep.training.config import BERT_CONFIG, SUPERVISED_TRAINING_ARGS
from bertrand.model.model import BERTrand
from bertrand.model.tokenization import tokenizer
from bertrand.training.evaluate import get_last_ckpt, load_metrics_df
from bertrand.training.evaluate import get_predictions


def parse_args() -> argparse.Namespace:
    """
    argument parser for the script
    :return: args
    """
    parser = argparse.ArgumentParser(
        "This script performs cross-validation for BERTrand"
    )
    # parser.add_argument(
    #     "--input-dir", type=str, required=True, help="Path to peptide:TCR dataset",
    # )

    parser.add_argument(
        "--model-ckpt",
        type=str,
        default=None,
        help="Path to model pre-trained checkpoint (omit for random init)",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Path to save weights and predictions.\n"
             "A subdirectory will be created for every csv.gz file in `input-dir`, "
             "every CV split will be in a separate subdirectory.",
    )

    parser.add_argument(
        "--n-splits", type=int, default=21, help="Number of CV splits",
    )

    return parser.parse_args()


def get_training_args(output_dir: str) -> TrainingArguments:
    """
    Returns pytorch-lightning training args.
    :param output_dir: folder to save checkpoints
    :return: training args
    """
    training_args = TrainingArguments(
        output_dir=output_dir, **SUPERVISED_TRAINING_ARGS,
    )
    return training_args


def train(
        train_dataset: TCRDataset,
        val_dataset: TCRDataset,
        model_class,
        model_ckpt: str,
        output_dir: str,
) -> dict:
    """
    Trains and evaluates the model.
    Returns predictions for the whole `val_dataset`, but computes metrics only for `split=='val'`,
    :param train_dataset: training set for the model
    :param val_dataset: validation and tests sets
    :param model_class: model class
    :param model_ckpt: model checkpoint (see train_mlm.py). if None, then weights are initialized randomly
    :param output_dir: folder to save model checkpoints and predictions for `val_dataset` for every epoch
    """
    predictions = []
    logging.info(f"Model class: {model_class}")

    best_checkpoint_dst = os.path.join(output_dir, f"best-checkpoint")
    if os.path.isdir(best_checkpoint_dst):
        last_checkpoint = get_last_ckpt(output_dir)
        metrics = load_metrics_df(last_checkpoint)
        metrics_eval = metrics[~metrics.eval_loss.isna()].copy()
        metrics_eval.epoch = metrics_eval.epoch.astype(int) - 1
        # metrics_eval.set_index("epoch", inplace=True)
        metrics_eval.to_csv(os.path.join(output_dir, 'metrics.csv'))
        best_epoch = metrics_eval.loc[metrics_eval.eval_roc.idxmax()]
        model = BERTrand.from_pretrained(best_checkpoint_dst)
        return dict(
            model=model,
            predictions=predictions,
            output_dir=output_dir,
            best_epoch=best_epoch,
        )

    def compute_metrics_and_save_predictions(eval_prediction):
        predictions.append(eval_prediction)
        preds = eval_prediction.predictions
        probs = torch.nn.functional.softmax(torch.tensor(preds), 1).numpy()[:, 1]
        labels = eval_prediction.label_ids
        roc = roc_auc_score(labels, probs)
        return {"roc": roc}

    if model_ckpt:
        logging.info(f"Loading model from {model_ckpt}")
        model = model_class.from_pretrained(model_ckpt)
        # for param in model.bert.parameters():
        #     param.requires_grad = False

    else:
        raise NotImplementedError()
        logging.info("Initializing model from scratch")
        model = model_class(BERT_CONFIG)

    training_args = get_training_args(output_dir)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics_and_save_predictions,
    )

    trainer.train()
    pd.to_pickle(predictions, os.path.join(output_dir, "predictions.pkl"))

    last_checkpoint = get_last_ckpt(output_dir)
    metrics = load_metrics_df(last_checkpoint)
    metrics_eval = metrics[~metrics.eval_loss.isna()].copy()
    metrics_eval.epoch = metrics_eval.epoch.astype(int) - 1
    metrics_eval.to_csv(os.path.join(output_dir, 'metrics.csv'))
    # metrics_eval.set_index("epoch", inplace=True)
    best_epoch = metrics_eval.loc[metrics_eval.eval_roc.idxmax()]
    print('Best epoch', best_epoch.to_dict())
    best_checkpoint = os.path.join(output_dir, f"checkpoint-{int(best_epoch.step)}")
    shutil.copytree(best_checkpoint, best_checkpoint_dst, dirs_exist_ok=True)
    model = BERTrand.from_pretrained(best_checkpoint)

    return dict(
        model=model,
        predictions=predictions,
        output_dir=output_dir,
        best_epoch=best_epoch,
    )


if __name__ == "__main__":
    from bertrand.immrep.data_sources import read_test, read_train_be

    logging.basicConfig(level=logging.INFO)

    args = parse_args()

    train_set = read_train_be()
    test_set = read_test()

    results = []
    for data_split in train_test_generator(train_set, test_set, NTEST=args.n_splits):
        pep_results = []
        y_pred_global = pd.Series(index=data_split['test_sample'].index, dtype=float)
        for pep, pep_df_train in data_split['train_sample'].groupby("Peptide", sort=False):
            pep_df_test = data_split['test_sample'].query(f'Peptide == "{pep}"')

            pep_df_val = pep_df_train.groupby(['y']).sample(frac=0.1)
            pep_df_train_proper = pep_df_train[~pep_df_train.index.isin(pep_df_val.index)]

            X_train = pep_df_train_proper[["CDR3a_extended", "CDR3b_extended", 'y']].reset_index(drop=True)
            X_val = pep_df_val[["CDR3a_extended", "CDR3b_extended", 'y']].reset_index(drop=True)
            X_test = pep_df_test[["CDR3a_extended", "CDR3b_extended", 'y']].reset_index(drop=True)

            # X_train = X_train.sample(4*32).reset_index(drop=True)
            # X_val = X_val.sample(2 * 32).reset_index(drop=True)
            # X_test = X_test.sample(3 * 32).reset_index(drop=True)

            train_dataset = TCRDataset(X_train)
            val_dataset = TCRDataset(X_val)
            test_dataset = TCRDataset(X_test)

            subdir_name = f"{pep}/test={data_split['test_iteration']}_train={data_split['train_iteration']}"
            out_dir = os.path.join(args.output_dir, subdir_name)
            logging.info(f"Saving weights and predictions to {out_dir}")
            train_result = train(train_dataset, val_dataset, BERTrand, args.model_ckpt, out_dir)
            test_predictions = get_predictions(train_result['model'], test_dataset)
            y_pred_test = torch.nn.functional.softmax(torch.tensor(test_predictions.predictions), 1).numpy()[:, 1]
            roc = roc_auc_score(X_test.y, y_pred_test)
            result_row = train_result['best_epoch']
            result_row['Peptide'] = pep
            result_row['test_iteration'] = data_split['test_iteration']
            result_row['train_iteration'] = data_split['train_iteration']
            result_row['model'] = 'BERTrand(pep)'
            result_row['roc'] = roc
            pep_results.append(result_row)
            y_pred_global.loc[pep_df_test.index] = y_pred_test
            print(f"Peptide {pep} AUC={roc:.3f}")

        not_na = ~y_pred_global.isna()
        roc_global = roc_auc_score(data_split['test_sample'].y[not_na], y_pred_global[not_na])
        print('Global AUC ', roc_global)
        for r in pep_results:
            r['roc_global'] = roc_global
            results.append(r)

    results_df = pd.concat(results, axis=1).T
    results_df.to_csv(os.path.join(args.output_dir, 'results.csv'))