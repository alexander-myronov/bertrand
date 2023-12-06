from sklearn.metrics import roc_auc_score


def pAUROC(X_test, probs):
    roc_pep = X_test.copy().assign(y_pred=probs).groupby('Peptide').apply(
        lambda x: roc_auc_score(x.y, x.y_pred, max_fpr=0.1))
    return roc_pep


def mean_pAUROC(X_test, probs):
    return pAUROC(X_test, probs).mean()
