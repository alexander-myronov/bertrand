import numpy as np
import pandas as pd

from Levenshtein import distance
from matplotlib import pyplot as plt
from scipy.spatial.distance import cdist

from bertrand.model.tokenization import AA_list

cdrcols = ["CDR3a_extended", "CDR3b_extended"]


def dist(x, y):
    return distance(x[0], y[0])


def sample_test(train, test, ratio=5, frac=0.05, fracneg=0.16, filter_lev=True, seed=42, verbose=False):
    # train = train_vdjdb
    np.random.seed(seed)

    l = test.CDR3a_extended.str.len()
    CDRA_MIN, CDRA_MAX = l.min(), l.max()
    l = test.CDR3b_extended.str.len()
    CDRB_MIN, CDRB_MAX = l.min(), l.max()
    if verbose:
        print('len(CDR3a) in', [CDRA_MIN, CDRA_MAX], 'len(CDR3n) in ', [CDRB_MIN, CDRB_MAX])

    pos = train[["Peptide", "CDR3b_extended", "CDR3a_extended"]].drop_duplicates()
    pos = pos[pos.Peptide.isin(test.Peptide)]
    vc = pos["Peptide"].value_counts()
    peps = vc.index[vc >= 10]
    pos = pos[pos.Peptide.isin(peps)]
    pos["y"] = 1

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

    # pos.shape
    # Tpos = 70
    # Tneg = 120

    potential_tcr_negatives = (
        pos.groupby("Peptide")
        .apply(lambda x: x.sample(frac=fracneg) if len(x) >= 10 / fracneg else x)
        .reset_index(drop=True)
    )
    # potential_tcr_negatives = pos.groupby("Peptide").apply(lambda x: x.sample(frac=fracneg)).reset_index(drop=True)

    # potential_tcr_negatives = pos.groupby("Peptide").apply(lambda x: x.sample(Tneg, replace=False) if len(x) >= Tneg else x).reset_index(drop=True)

    peps = pd.Series(peps, index=peps)
    rlist = []

    for i, pep in enumerate(peps.index):
        pep_positives = pos[pos.Peptide == pep].copy()
        pos_i = pd.MultiIndex.from_frame(pep_positives[cdrcols])

        dist_over_3 = peps.apply(lambda p: distance(p, pep) > 3)
        pep_potential_negatives = peps[dist_over_3]

        train_neg = potential_tcr_negatives[potential_tcr_negatives.Peptide.isin(pep_potential_negatives)]
        train_neg = train_neg[cdrcols].drop_duplicates()
        train_neg_i = pd.MultiIndex.from_frame(train_neg)
        train_neg = train_neg[~train_neg_i.isin(pos_i)]

        if filter_lev:
            adist = cdist(
                train_neg.CDR3a_extended.values.reshape(-1, 1),
                pep_positives.CDR3a_extended.values.reshape(-1, 1),
                metric=dist,
            )
            bdist = cdist(
                train_neg.CDR3b_extended.values.reshape(-1, 1),
                pep_positives.CDR3b_extended.values.reshape(-1, 1),
                metric=dist,
            )
            adist, bdist = adist.min(axis=1), bdist.min(axis=1)
            # print(adist)
            # print(adist.shape)
            # print(train_neg.shape[0])
            train_neg = train_neg[(adist >= 3) & (bdist >= 3)]
            # print(train_neg.shape[0])

        if len(pep_positives) * frac > 10:
            print(pep, len(pep_positives), frac)
            pep_positives = pep_positives.sample(frac=frac, replace=False)
        # else:
            # continue
        # if len(pep_positives) > 70:
        #     pep_positives = pep_positives.sample(70)

        # if len(pep_positives) > Tpos:
        #     pep_positives = pep_positives.sample(Tpos, replace=False)
        if verbose:
            print(pep, len(pep_positives), len(train_neg))
        sample_neg = train_neg.sample(min(len(pep_positives) * ratio, len(train_neg)), replace=False)
        sample_neg["Peptide"] = pep
        sample_neg["y"] = 0
        # print(pep, len(pep_positives), len(sample_neg))

        rlist.append(pd.concat([pep_positives, sample_neg]))
    df = pd.concat(rlist)
    df = df.groupby(["Peptide"] + cdrcols).y.agg(max).reset_index()
    m = df.groupby(["CDR3a_extended", "CDR3b_extended", "Peptide"]).y.agg("nunique").unique()
    assert set(m) == {1}
    return df

from Levenshtein import distance

train_sample_peps = [
    "TPRVTGGGAM",
    "RPPIFIRRL",
    "IVTDFSVIK",
    "RAKFKQLL",
    "GILGFVFTL",
    "YLQPRTFLL",
    "NLVPMVATV",
    "VTEHDTLLY",
    "RPHERNGFTVL",
    "GLCTLVAML",
]


def missing(train, test):
    test_peps = test.Peptide.drop_duplicates()
    totally_missing_peptides = test_peps[~test_peps.isin(train.Peptide)]

    rate_peptides = np.setdiff1d(np.setdiff1d(test_peps, train_sample_peps), totally_missing_peptides)

    return list(rate_peptides), list(totally_missing_peptides)


def sample_additional(train, test):
    substitute_peptides = []
    peps = train.Peptide.drop_duplicates()
    peps = peps[~peps.isin(test.Peptide)]
    rare_peptides, totally_missing_peptides = missing(train, test)
    for tmp in rare_peptides + totally_missing_peptides:
        # print("%%%\n", tmp)
        d = np.array([distance(p1, tmp) for p1 in peps])
        # raise Exception()
        similar_peps = peps.iloc[np.argsort(np.array(d))]
        similar_peps = similar_peps[~similar_peps.isin(test.Peptide)]
        similar_peps = pd.Series(np.sort(d), index=similar_peps.values)
        similar_peps = similar_peps[similar_peps <= 7]

        counts = train.Peptide.value_counts().loc[similar_peps.index]

        c = pd.concat([similar_peps, counts], axis=1)
        c.columns = ["dist", "count"]
        c = c[(c["count"] >= 11) & (c["count"] <= 20)]
        # print(c)
        best_substitute = c.sample().index[0]
        # print(tmp, best_substitute)
        substitute_peptides.append(best_substitute)
    return list(substitute_peptides), list(rare_peptides)

def sample_test_additional(train, test, ratio=5, frac=0.05, fracneg=0.16, filter_lev=True, seed=42,
                verbose=False):
    # train = train_vdjdb
    np.random.seed(seed)

    substitute_peptides, rare_peptides = sample_additional(train, test)
    additional_peptides = substitute_peptides + rare_peptides
    # print('additional', additional_peptides)
    l = test.CDR3a_extended.str.len()
    CDRA_MIN, CDRA_MAX = l.min(), l.max()
    l = test.CDR3b_extended.str.len()
    CDRB_MIN, CDRB_MAX = l.min(), l.max()
    if verbose:
        print('len(CDR3a) in', [CDRA_MIN, CDRA_MAX], 'len(CDR3n) in ', [CDRB_MIN, CDRB_MAX])

    pos = train[["Peptide", "CDR3b_extended", "CDR3a_extended"]].drop_duplicates()

    pos = pos[pos.Peptide.isin(test.Peptide) | pos.Peptide.isin(additional_peptides)]
    vc = pos["Peptide"].value_counts()
    peps = vc.index[vc >= 10]
    pos = pos[pos.Peptide.isin(peps)]
    pos["y"] = 1
    # print(pos.Peptide.value_counts())
    # raise Exception()

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

    # pos.shape
    # Tpos = 70
    # Tneg = 120

    potential_tcr_negatives = (
        pos.groupby("Peptide")
        .apply(lambda x: x.sample(frac=fracneg) if len(x) >= 10 / fracneg else x)
        .reset_index(drop=True)
    )
    # potential_tcr_negatives = pos.groupby("Peptide").apply(lambda x: x.sample(frac=fracneg)).reset_index(drop=True)

    # potential_tcr_negatives = pos.groupby("Peptide").apply(lambda x: x.sample(Tneg, replace=False) if len(x) >= Tneg else x).reset_index(drop=True)

    peps = pd.Series(peps, index=peps)
    rlist = []
    pos_gb = pos.groupby('Peptide')
    for i, pep in enumerate(peps.index):
        pep_positives = pos_gb.get_group(pep).copy()
        pos_i = pd.MultiIndex.from_frame(pep_positives[cdrcols])

        dist_over_3 = peps.apply(lambda p: distance(p, pep) > 3)
        pep_potential_negatives = peps[dist_over_3]

        train_neg = potential_tcr_negatives[potential_tcr_negatives.Peptide.isin(pep_potential_negatives)]
        train_neg = train_neg[cdrcols].drop_duplicates()
        train_neg_i = pd.MultiIndex.from_frame(train_neg)
        train_neg = train_neg[~train_neg_i.isin(pos_i)]

        if filter_lev:
            adist = cdist(
                train_neg.CDR3a_extended.values.reshape(-1, 1),
                pep_positives.CDR3a_extended.values.reshape(-1, 1),
                metric=dist,
            )
            bdist = cdist(
                train_neg.CDR3b_extended.values.reshape(-1, 1),
                pep_positives.CDR3b_extended.values.reshape(-1, 1),
                metric=dist,
            )
            adist, bdist = adist.min(axis=1), bdist.min(axis=1)
            # print(adist)
            # print(adist.shape)
            # print(train_neg.shape[0])
            train_neg = train_neg[(adist >= 3) & (bdist >= 3)]
            # print(train_neg.shape[0])

        if len(pep_positives) * frac > 10:
            print(pep, len(pep_positives), frac)
            pep_positives = pep_positives.sample(frac=frac, replace=False)
        # else:
        # continue
        # if len(pep_positives) > 70:
        #     pep_positives = pep_positives.sample(70)

        # if len(pep_positives) > Tpos:
        #     pep_positives = pep_positives.sample(Tpos, replace=False)
        if verbose:
            print(pep, len(pep_positives), len(train_neg))
        sample_neg = train_neg.sample(min(len(pep_positives) * ratio, len(train_neg)), replace=False)
        sample_neg["Peptide"] = pep
        sample_neg["y"] = 0
        # print(pep, len(pep_positives), len(sample_neg))

        rlist.append(pd.concat([pep_positives, sample_neg]))
    df = pd.concat(rlist)
    df = df.groupby(["Peptide"] + cdrcols).y.agg(max).reset_index()
    m = df.groupby(["CDR3a_extended", "CDR3b_extended", "Peptide"]).y.agg("count").unique()
    assert set(m) == {1}
    return df


def diagnostics(train, synthetic_test, test):
    fig, axs = plt.subplots(2, 2)

    synthetic_test[["CDR3a_extended", "CDR3b_extended"]].value_counts().plot(ax=axs[0, 0])
    axs[0, 0].set_title('Synthetic test set')
    synthetic_test[["CDR3a_extended", "CDR3b_extended"]].value_counts().hist(ax=axs[0, 1])
    axs[0, 1].set_title('Synthetic test set')

    vc = train.Peptide.value_counts()
    possible_test = test[test.Peptide.isin(vc[vc >= 10].index)]
    possible_test[["CDR3a_extended", "CDR3b_extended"]].value_counts().plot(ax=axs[1, 0])
    axs[1, 0].set_title('Possible test set')
    possible_test[["CDR3a_extended", "CDR3b_extended"]].value_counts().hist(ax=axs[1, 1])
    axs[1, 1].set_title('Possible test set')


def plot_length(train, test, col):
    N = 10
    fig, axs = plt.subplots(nrows=N // 5, ncols=5)
    axs = axs.flat
    peps = train[train.Peptide.isin(test.Peptide)].Peptide.value_counts().index
    for i, p in enumerate(peps[:N]):
        ax = axs[i]
        t = test[test.Peptide == p]
        t[col].str.len().hist(ax=ax, density=True, alpha=0.7, label=f"{col} test", bins=np.arange(9, 22))
        # t.CDR3b_extended.str.len().hist(ax=ax, alpha=0.7, label='beta test')

        tr = train[train.Peptide == p]
        tr[col].str.len().hist(ax=ax, density=True, alpha=0.7, label=f"{col} train", bins=np.arange(9, 22))
        # tr.CDR3b_extended.str.len().hist(ax=ax, alpha=0.7, label='beta train')
        if i == 0:
            ax.legend()
        ax.set_title(f"{p} {len(t)} {len(tr)}")

    fig.set_size_inches(16, 9)


if __name__ == '__main__':
    from bertrand.immrep.data_sources import read_test, read_train_be

    test = read_test()
    train = read_train_be()

    synthetic_test = sample_test(
        train,
        test,
        frac=0.05,
        fracneg=0.16)

    diagnostics(train, synthetic_test, test)
    plot_length(synthetic_test, test, "CDR3a_extended")
    plot_length(synthetic_test, test, "CDR3b_extended")
    plt.show()
    # synthetic_test = sample_test(
    #     train_vdjdb,
    #     test,
    #     frac=0.05,
    #     fracneg=0.16,
    # )
