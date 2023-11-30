import numpy as np
import pandas as pd
from torch.utils.data import Dataset

from bertrand.model.tokenization import tokenizer
from transformers.tokenization_utils_base import BatchEncoding


class PeptideTCRDataset(Dataset):
    def __init__(self, dataset: pd.DataFrame):
        self.examples = dataset

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        row = self.examples.iloc[i]
        ab = tokenizer.encode_plus(
            " ".join(list(row.CDR3a_extended)), " ".join(list(row.CDR3b_extended))
        )
        pep = tokenizer.encode_plus(" ".join(list(row.Peptide)))
        pep['input_ids'] = pep['input_ids'] + ab['input_ids'][1:]
        pep['token_type_ids'] = pep['token_type_ids'] + [v + 1 for v in ab['token_type_ids'][1:]]
        pep['attention_mask'] = pep['attention_mask'] + ab['attention_mask'][1:]
        if "y" in self.examples.columns:
            pep.data["labels"] = row.y
        if "weight" in self.examples.columns:
            pep.data["weights"] = row.weight
        return pep

    def calc_weights(self) -> None:
        """
        Calculates weights for peptide:TCR prediction problem.
        The weight of an observation depends on it's peptide and TCR abundance in the dataset
        Popular peptide and TCR clusters are downweighted
        """
        pep_count = self.examples.Peptide.value_counts()
        pep_w = pep_count.loc[self.examples.Peptide].values
        # 1 / pep_w gives very low weights for popular peptide clusters
        pep_w = 1 / np.log(2 + pep_w)

        tcra_count = self.examples.CDR3a_extended.value_counts()
        tcra_w = tcra_count.loc[self.examples.CDR3a_extended].values
        tcra_w = 1 / np.log(2 + tcra_w)

        tcrb_count = self.examples.CDR3a_extended.value_counts()
        tcrb_w = tcrb_count.loc[self.examples.CDR3a_extended].values
        tcrb_w = 1 / np.log(2 + tcrb_w)

        weights = pep_w * 10 * (tcra_w + tcrb_w)
        self.examples.loc[:, "weight"] = weights
