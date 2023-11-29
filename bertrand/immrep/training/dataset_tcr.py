import pandas as pd
from torch.utils.data import Dataset

from bertrand.model.tokenization import tokenizer


class TCRDataset(Dataset):
    def __init__(self, dataset: pd.DataFrame):
        self.examples = dataset

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        row = self.examples.iloc[i]
        encode_plus = tokenizer.encode_plus(
            " ".join(list(row.CDR3a_extended)), " ".join(list(row.CDR3b_extended))
        )
        if "y" in self.examples.columns:
            encode_plus.data["labels"] = row.y
        return encode_plus
