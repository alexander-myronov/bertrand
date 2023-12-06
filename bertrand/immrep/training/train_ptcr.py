import argparse
import logging
import os
import shutil

import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from transformers import TrainingArguments, DataCollatorWithPadding, Trainer

from bertrand.immrep.sample_test_set import train_sample_peps
from bertrand.immrep.training.cv import train_test_additional_generator
from bertrand.immrep.training.dataset_ptcr import PeptideTCRDataset
from bertrand.immrep.training.dataset_tcr import TCRDataset
from bertrand.immrep.training.config import BERT_CONFIG, SUPERVISED_TRAINING_ARGS_PTCR
from bertrand.immrep.training.metrics import mean_pAUROC, pAUROC
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
        output_dir=output_dir, **SUPERVISED_TRAINING_ARGS_PTCR,
    )
    return training_args


def train(
        train_dataset: PeptideTCRDataset,
        val_dataset: PeptideTCRDataset,
        test_dataset: PeptideTCRDataset,
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
        best_epoch = metrics_eval.loc[metrics_eval.eval_pauroc_val.idxmax()]
        model = BERTrand.from_pretrained(best_checkpoint_dst)
        return dict(
            model=model,
            predictions=predictions,
            output_dir=output_dir,
            best_epoch=best_epoch,
        )

    joint_examples = pd.concat([
        val_dataset.examples.assign(split='val'),
        test_dataset.examples.assign(split='test')
    ]).reset_index(drop=True)
    joint_dataset = PeptideTCRDataset(joint_examples)

    def compute_metrics_and_save_predictions(eval_prediction):
        predictions.append(eval_prediction)
        preds = eval_prediction.predictions
        probs = torch.nn.functional.softmax(torch.tensor(preds), 1).numpy()[:, 1]
        labels = eval_prediction.label_ids
        val_mask = joint_examples.split == 'val'
        roc_val = roc_auc_score(labels[val_mask], probs[val_mask])
        pauroc_val = mean_pAUROC(joint_dataset.examples[val_mask], probs[val_mask])
        test_mask = joint_examples.split == 'test'
        roc_test = roc_auc_score(labels[test_mask], probs[test_mask])
        pauroc_test = mean_pAUROC(joint_dataset.examples[test_mask], probs[test_mask])
        test_iso_mask = (joint_examples.split == 'test') & (~joint_examples.Peptide.isin(train_sample_peps))
        roc_test_iso = roc_auc_score(labels[test_iso_mask], probs[test_iso_mask])
        pauroc_test_iso = mean_pAUROC(joint_dataset.examples[test_iso_mask], probs[test_iso_mask])
        return {"roc_val": roc_val,
                "roc_test": roc_test,
                "roc_test_iso": roc_test_iso,
                'pauroc_val': pauroc_val,
                'pauroc_test': pauroc_test,
                'pauroc_test_iso': pauroc_test_iso}

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
        eval_dataset=joint_dataset,
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
    best_epoch = metrics_eval.loc[metrics_eval.eval_pauroc_val.idxmax()]
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
    data_splits = pd.read_pickle(
        '/home/ardigen/Documents/bertrand/bertrand/notebooks/immrep/data_splits_additional7.pkl')
    # data_splits = train_test_additional_generator(train_set, test_set, NTEST=args.n_splits)

    for data_split in data_splits:
        data_split['test_sample'] = data_split['test_sample'].sample(1000, random_state=42).reset_index(drop=True)
        data_split['train_sample'] = data_split['train_sample'].sample(1000, random_state=42).reset_index(drop=True)

        X = data_split['train_sample'].reset_index(drop=True)
        X_test = data_split['test_sample'].reset_index(drop=True)
        X_val = X.groupby(['Peptide', 'y']).sample(frac=0.1)
        X_train = X[~X.index.isin(X_val.index)]

        # X_train = X_train.sample(4 * 32).reset_index(drop=True)
        # X_val = X_val.sample(2 * 32).reset_index(drop=True)
        # X_test = X_test.sample(3 * 32).reset_index(drop=True)

        train_dataset = PeptideTCRDataset(X_train)
        val_dataset = PeptideTCRDataset(X_val)
        test_dataset = PeptideTCRDataset(X_test)

        subdir_name = f"test={data_split['test_iteration']}_train={data_split['train_iteration']}"
        out_dir = os.path.join(args.output_dir, subdir_name)
        logging.info(f"Saving weights and predictions to {out_dir}")
        train_result = train(train_dataset, val_dataset, test_dataset, BERTrand, args.model_ckpt, out_dir)
        test_predictions = get_predictions(train_result['model'], test_dataset)
        y_pred_test = torch.nn.functional.softmax(torch.tensor(test_predictions.predictions), 1).numpy()[:, 1]
        roc = roc_auc_score(X_test.y, y_pred_test)
        mean_pauroc = mean_pAUROC(X_test, y_pred_test)

        m = ~X_test.Peptide.isin(train_sample_peps)
        roc_iso = roc_auc_score(X_test[m].y, y_pred_test[m])
        mean_pauroc_iso = mean_pAUROC(X_test[m], y_pred_test[m])
        result_row = train_result['best_epoch']
        result_row['test_iteration'] = data_split['test_iteration']
        result_row['train_iteration'] = data_split['train_iteration']
        result_row['model'] = 'BERTrand(pan)'
        result_row['mean_pauroc'] = mean_pauroc
        result_row['mean_pauroc_iso'] = mean_pauroc_iso
        result_row['roc'] = roc
        result_row['roc_iso'] = roc_iso

        print(f"pAUC={mean_pauroc:.3f} pAUCiso={mean_pauroc_iso:.3f} AUC={roc:.3f} AUCiso={roc_iso:.3f} ")
        print(pAUROC(X_test, y_pred_test))

        results.append(result_row)

    results_df = pd.concat(results, axis=1).T
    results_df.to_csv(os.path.join(args.output_dir, 'results.csv'))
