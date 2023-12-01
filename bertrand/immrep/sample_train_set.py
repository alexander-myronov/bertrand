import numpy as np
import pandas as pd
from Levenshtein import distance
from scipy.spatial.distance import cdist
from bertrand.immrep.sample_test_set import dist, missing
from bertrand.model.tokenization import AA_list


def filter_cdr(pos, synthetic_test, t=2):
    adist = cdist(
        pos.CDR3a_extended.values.reshape(-1, 1), synthetic_test.CDR3a_extended.values.reshape(-1, 1), metric=dist
    )
    bdist = cdist(
        pos.CDR3b_extended.values.reshape(-1, 1), synthetic_test.CDR3b_extended.values.reshape(-1, 1), metric=dist
    )

    adist, bdist = adist.min(axis=1), bdist.min(axis=1)

    # (adist >= 2).sum()
    # (bdist >= 2).sum()
    # ((adist >= 2) & (bdist >= 2)).sum()
    return pos[(adist >= t) & (bdist >= t)]


def sample_train(train, test, synthetic_test, T=50, seed=42, ratio=5, include_other_peptides=False, verbose=False):
    np.random.seed(seed)
    l = test.CDR3a_extended.str.len()
    CDRA_MIN, CDRA_MAX = l.min(), l.max()
    l = test.CDR3b_extended.str.len()
    CDRB_MIN, CDRB_MAX = l.min(), l.max()
    if verbose:
        print('len(CDR3a) in', [CDRA_MIN, CDRA_MAX], 'len(CDR3b) in ', [CDRB_MIN, CDRB_MAX])

    pos = train[["Peptide", "CDR3b_extended", "CDR3a_extended"]].drop_duplicates()
    pos["y"] = 1

    assert set([pos.groupby(["Peptide", "CDR3b_extended", "CDR3a_extended"]).y.agg("count").max()]) == {1}

    if include_other_peptides:
        pos = pos[pos.Peptide.isin(test.Peptide) | pos.Peptide.isin(pos.Peptide.value_counts().iloc[1:].head(10).index)]
    else:
        pos = pos[pos.Peptide.isin(test.Peptide)]
    vc = pos["Peptide"].value_counts()
    peps = vc.index[vc >= T]
    pos = pos[pos.Peptide.isin(peps)]

    la = pos.CDR3a_extended.str.len()
    lb = pos.CDR3b_extended.str.len()
    pos = pos[
        (la >= CDRA_MIN)
        & (la <= CDRA_MAX)
        & (lb >= CDRB_MIN)
        & (lb <= CDRB_MAX)
        & (pos.CDR3a_extended.str[0] == "C")
        & (pos.CDR3b_extended.str[0] == "C")
        ]

    pos_filt = filter_cdr(pos, synthetic_test)
    if verbose:
        print(f"{len(pos)} positive observations total")
        print(f"{len(pos_filt)} positive observations filtered")
        print(pos_filt.Peptide.value_counts())

    rlist = []
    ppeps = []
    for pep in pos_filt.Peptide.value_counts().index:

        pos_pep = pos_filt[pos_filt.Peptide == pep]
        if pep not in test.Peptide.unique() or len(pos_pep) < T:
            continue

        potential_negs = pos_filt[pos_filt.Peptide != pep]
        potential_negs_filt = filter_cdr(potential_negs, pos_pep, t=2).copy()
        if verbose:
            print(pep, len(pos_pep) * ratio, len(potential_negs), len(potential_negs_filt))
        if len(pos_pep) * ratio >= len(potential_negs_filt):
            neg_sample = potential_negs_filt.copy()
        else:
            neg_sample = potential_negs_filt.sample(len(pos_pep) * ratio)
        neg_sample["y"] = 0
        neg_sample["Peptide"] = pep
        rlist.append(neg_sample)
        ppeps.append(pep)

    df = pd.concat([pos_filt[pos_filt.Peptide.isin(ppeps)]] + rlist)
    df = df.groupby(["Peptide", "CDR3b_extended", "CDR3a_extended"]).y.max().reset_index()
    assert {df.groupby(["Peptide", "CDR3b_extended", "CDR3a_extended"]).y.agg("nunique").max()} == {1}
    return df

def sample_additional(train, test, test_sample):
    peps = train.Peptide.drop_duplicates()
    peps = np.setdiff1d(peps, test.Peptide)
    peps = np.setdiff1d(peps, test_sample.Peptide)
    missing_peptides, totally_missing_peptides = missing(train, test)

    d = np.array([min([distance(p1, tmp) for tmp in missing_peptides+totally_missing_peptides]) for p1 in peps])
    # raise Exception()
    similar_peps = pd.Series(peps[np.argsort(np.array(d))])
    # similar_peps = similar_peps[~similar_peps.isin(test.Peptide)]
    similar_peps = pd.Series(np.sort(d), index=similar_peps.values)
    similar_peps = similar_peps[similar_peps <= 10]

    counts = train.Peptide.value_counts().loc[similar_peps.index]

    c = pd.concat([similar_peps, counts], axis=1)
    c.columns = ["dist", "count"]
    c = c[(c["count"] >= 10)]
    addition_train_peptides = list(c.index)
    addition_train_peptides.remove('KLGGALQAK')
    return addition_train_peptides
    # raise Exception()

def sample_train_additional(train, test, synthetic_test, T=10, seed=42, ratio=5, verbose=False):
    np.random.seed(seed)
    l = test.CDR3a_extended.str.len()
    CDRA_MIN, CDRA_MAX = l.min(), l.max()
    l = test.CDR3b_extended.str.len()
    CDRB_MIN, CDRB_MAX = l.min(), l.max()
    if verbose:
        print("len(CDR3a) in", [CDRA_MIN, CDRA_MAX], "len(CDR3b) in ", [CDRB_MIN, CDRB_MAX])

    pos = train[["Peptide", "CDR3b_extended", "CDR3a_extended"]].drop_duplicates()
    pos["y"] = 1

    assert {pos.groupby(["Peptide", "CDR3b_extended", "CDR3a_extended"]).y.agg("count").max()} == {1}

    additional_peptides = sample_additional(train, test, synthetic_test)
    # if include_other_peptides:
    pos = pos[pos.Peptide.isin(synthetic_test.Peptide) | pos.Peptide.isin(additional_peptides)]
    # else:
    # pos = pos[pos.Peptide.isin(test.Peptide)]
    vc = pos["Peptide"].value_counts()
    peps = vc.index[vc >= T]
    pos = pos[pos.Peptide.isin(peps)]

    la = pos.CDR3a_extended.str.len()
    lb = pos.CDR3b_extended.str.len()
    regex = f"[{''.join(AA_list)}]+"
    pos = pos[
        (la >= CDRA_MIN)
        & (la <= CDRA_MAX)
        & (lb >= CDRB_MIN)
        & (lb <= CDRB_MAX)
        & (pos.CDR3a_extended.str[0] == "C")
        & (pos.CDR3b_extended.str[0] == "C")
        & pos.CDR3a_extended.str.fullmatch(regex)
        & pos.CDR3b_extended.str.fullmatch(regex)
    ]

    pos_filt = filter_cdr(pos, synthetic_test)

    # potential_tcr_negatives = (
    #     pos_filt.groupby("Peptide")
    #     .apply(lambda x: x.sample(frac=0.2) if len(x) >= 10 / 0.2 else x)
    #     .reset_index(drop=True)
    # )

    potential_tcr_negatives = (
        pos_filt.groupby("Peptide")
        .apply(lambda x: x.sample(500) if len(x) >= 500 else x)
        .reset_index(drop=True)
    )
    # print(pos_filt.Peptide.value_counts())
    # raise Exception()
    if verbose:
        print(f"{len(pos)} positive observations total")
        print(f"{len(pos_filt)} positive observations filtered")
        print(f"{len(potential_tcr_negatives)} potential negatives  filtered")
        print(pos_filt.Peptide.value_counts())

    rlist = []
    plist = []
    pos_filt_gb = pos_filt.groupby('Peptide')
    for pep, pos_pep in pos_filt_gb:
        if len(pos_pep) < T:
            continue
        if len(pos_pep) > 500:
            pos_pep = pos_pep.sample(500).reset_index(drop=True)

        potential_negs = potential_tcr_negatives[potential_tcr_negatives.Peptide != pep]
        potential_negs_filt = filter_cdr(potential_negs, pos_pep, t=2).copy()

        if verbose:
            print(pep, len(pos_pep) * ratio, len(potential_negs), len(potential_negs_filt))
        if len(pos_pep) * ratio >= len(potential_negs_filt):
            neg_sample = potential_negs_filt.copy()
        else:
            neg_sample = potential_negs_filt.sample(len(pos_pep) * ratio)
        neg_sample["y"] = 0
        neg_sample["Peptide"] = pep
        rlist.append(neg_sample)
        plist.append(pos_pep)

    df = pd.concat(plist+rlist)
    df = df.groupby(["Peptide", "CDR3b_extended", "CDR3a_extended"]).y.max().reset_index()
    assert {df.groupby(["Peptide", "CDR3b_extended", "CDR3a_extended"]).y.agg("nunique").max()} == {1}
    return df


if __name__ == '__main__':
    from bertrand.immrep.data_sources import read_train_be, read_test
    from bertrand.immrep.sample_test_set import sample_test
    test = read_test()
    train = read_train_be()
    synthetic_test = sample_test(train, test, frac=0.07, fracneg=0.17, seed=43)
    train_df = sample_train(train, test, synthetic_test, T=50)
    print('Sampling done')
    print(pd.crosstab(train_df.Peptide, train_df.y))