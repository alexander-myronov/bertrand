import os

import pandas as pd

from bertrand.immrep.paths import DATA_DIR

def read_test():
    test = pd.read_csv(os.path.join(DATA_DIR, "immrep/test.csv"))
    return test


def read_train_immrep():
    train_immrep3 = pd.read_csv(os.path.join(DATA_DIR, "immrep/VDJdb_paired_chain.csv"))
    return train_immrep3

def read_train_be():
    be = pd.read_parquet(os.path.join(DATA_DIR, "binding-estimation"))
    binders = be[be.is_binder]
    alpha_beta_binders = binders[~binders.CDR3_beta.isna() & ~binders.CDR3_alpha.isna()]
    train_immuno = alpha_beta_binders.rename(
        columns={"CDR3_alpha": "CDR3a_extended", "CDR3_beta": "CDR3b_extended", "HLA": "HLA_pseudoseq",
                 "HLA_type": "HLA"}
    )
    return train_immuno

def read_train_vdjdb():
    train_vdjdb = pd.read_csv(os.path.join(DATA_DIR, "vdjdb/vdjdb_full.txt"), sep="\t")
    train_vdjdb = train_vdjdb.rename(
        columns={
            "cdr3.alpha": "CDR3a_extended",
            "cdr3.beta": "CDR3b_extended",
            "mhc.a": "HLA",
            "antigen.epitope": "Peptide",
        }
    )
    train_vdjdb = train_vdjdb[
        ~train_vdjdb.CDR3a_extended.isna()
        & ~train_vdjdb.CDR3b_extended.isna()
        & (train_vdjdb.species == "HomoSapiens")
        & (train_vdjdb["mhc.class"] == "MHCI")
        ]
    return train_vdjdb


def diagnostics(train, test):
    print(f"{len(train)} examples")
    m = train.CDR3a_extended.isin(test.CDR3a_extended) | train.CDR3b_extended.isin(test.CDR3b_extended)
    print(f"{(~m).sum()} usable examples, {m.sum()} in limbo")
    print("### CDR")
    print("Length disribution")
    a = train.CDR3a_extended.str.len().value_counts().sort_index()
    b = train.CDR3b_extended.str.len().value_counts().sort_index()
    j = pd.concat([a, b], axis=1).fillna(0).astype(int)
    j.index = j.index.astype(int)
    print(j)

    nad = len(train.CDR3a_extended.drop_duplicates())
    nbd = len(train.CDR3b_extended.drop_duplicates())
    nnd = len(train[["CDR3a_extended", "CDR3b_extended"]].drop_duplicates())

    print(f"{nad} CDR3a unique in {len(train)}")
    print(f"{nbd} CDR3b unique in {len(train)}")
    print(f"{nnd} unique CDR3a,CDR3b pairs in {len(train)}")

    print("### Peptide")
    pvc = train.Peptide.value_counts()
    print(f"{len(pvc)} unique peptides")
    for top in [5, 10, 15]:
        top5 = pvc.head(top).sum() / len(train)
        print(f"{top5:.2f} observations in top {top} peptides")

    print("Test set peptides present")
    print("Observations for test peptides")
    print(test.Peptide.drop_duplicates().isin(train.Peptide).value_counts().sort_index())
    td = train[train.Peptide.isin(test.Peptide)]

    print(td.Peptide.value_counts())

    cols = ["CDR3a_extended", "CDR3b_extended"]
    test_index = pd.MultiIndex.from_frame(test[cols])
    train_index = pd.MultiIndex.from_frame(train[cols])
    s = test_index.isin(train_index).sum()
    print(f"{s} exact test observations in train")

    print("### HLA")
    print(train.HLA.value_counts().head(10))


if __name__ == '__main__':
    test = read_test()
    train_vdjdb = read_train_vdjdb()
    train_be = read_train_be()
    train_immrep = read_train_immrep()
    # diagnostics(train, test)

    def get_test_counts(train, test):
        return train.Peptide[train.Peptide.isin(test.Peptide)].value_counts()  # .fillna(0).astype(int)


    stats = (
        pd.concat(
            [
                get_test_counts(train, test)
                for train in [train_immrep, train_be, train_vdjdb]
            ],
            axis=1,
        )
        .fillna(0)
        .astype(int)
    )
    stats.columns = ["immrep3", "be", "vdjdb"]
    print(stats)
