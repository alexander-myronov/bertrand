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
        pep['token_type_ids'] = pep['token_type_ids'] + [v+1 for v in ab['token_type_ids'][1:]]
        pep['attention_mask'] = pep['attention_mask'] + ab['attention_mask'][1:]
        if "y" in self.examples.columns:
            pep.data["labels"] = row.y
        return pep
