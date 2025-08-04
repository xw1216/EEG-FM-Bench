import bisect
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import sklearn.metrics as sklearn_metrics
from numpy import ndarray


def size(y_pred:np.ndarray):
    """Average size of the prediction set.
    """
    return np.mean(y_pred.sum(1))

def rejection_rate(y_pred:np.ndarray):
    """Rejection rate, defined as the proportion of samples with prediction set size != 1
    """
    return np.mean(y_pred.sum(1) != 1)

def _missrate(y_pred:np.ndarray, y_true:np.ndarray, ignore_rejected=False):
    """Computes the class-wise mis-coverage rate (or risk).

    Args:
        y_pred (np.ndarray): prediction scores.
        y_true (np.ndarray): true labels.
        ignore_rejected (bool, optional): If True, we compute the miscoverage rate
            without rejection  (that is, condition on the unrejected samples). Defaults to False.

    Returns:
        np.ndarray: miss-coverage rates for each class.
    """
    # currently handles multilabel and multiclass
    K = y_pred.shape[1]
    if len(y_true.shape) == 1:
        y_true, _ = np.zeros((len(y_true),K), dtype=bool), y_true
        y_true[np.arange(len(y_true)), _] = 1
    y_true = y_true.astype(bool)

    keep_msk = (y_pred.sum(1) == 1) if ignore_rejected else np.ones(len(y_true), dtype=bool)
    missed = []
    for k in range(K):
        missed.append(1-np.mean(y_pred[keep_msk & y_true[:, k], k]))

    return np.asarray(missed)


def miscoverage_ps(y_pred:np.ndarray, y_true:np.ndarray):
    """Miscoverage rates for all samples (similar to recall).

    Example:
        >>> y_pred = np.asarray([[1,0,0],[1,0,0],[1,1,0],[0, 1, 0]])
        >>> y_true = np.asarray([1,0,1,2])
        >>> error_ps(y_pred, y_true)
        array([0. , 0.5, 1. ])


    Explanation:
    For class 0, the 1-th prediction set ({0}) contains the label, so the miss-coverage is 0/1=0.
    For class 1, the 0-th prediction set ({0}) does not contain the label, the 2-th prediction
    set ({0,1}) contains the label. Thus, the miss-coverage is 1/2=0.5.
    For class 2, the last prediction set is {1} and the label is 2, so the miss-coverage is 1/1=1.
    """
    return _missrate(y_pred, y_true, False)

def error_ps(y_pred:np.ndarray, y_true:np.ndarray):
    """Miscoverage rates for unrejected samples, where rejection is defined to be sets with size !=1).

    Example:
        >>> y_pred = np.asarray([[1,0,0],[1,0,0],[1,1,0],[0, 1, 0]])
        >>> y_true = np.asarray([1,0,1,2])
        >>> error_ps(y_pred, y_true)
        array([0., 1., 1.])

    Explanation:
    For class 0, the 1-th sample is correct and not rejected, so the error is 0/1=0.
    For class 1, the 0-th sample is incorrerct and not rejected, the 2-th is rejected.
    Thus, the error is 1/1=1.
    For class 2, the last sample is not-rejected but the prediction set is {1}, so the error
    is 1/1=1.
    """
    return _missrate(y_pred, y_true, True)

def miscoverage_overall_ps(y_pred:np.ndarray, y_true:np.ndarray):
    """Miscoverage rate for the true label. Only for multiclass.

    Example:
        >>> y_pred = np.asarray([[1,0,0],[1,0,0],[1,1,0]])
        >>> y_true = np.asarray([1,0,1])
        >>> miscoverage_overall_ps(y_pred, y_true)
        0.333333

    Explanation:
    The 0-th prediction set is {0} and the label is 1 (not covered).
    The 1-th prediction set is {0} and the label is 0 (covered).
    The 2-th prediction set is {0,1} and the label is 1 (covered).
    Thus the miscoverage rate is 1/3.
    """
    assert len(y_true.shape) == 1
    truth_pred = y_pred[np.arange(len(y_true)), y_true]

    return 1 - np.mean(truth_pred)

def error_overall_ps(y_pred:np.ndarray, y_true:np.ndarray):
    """Overall error rate for the un-rejected samples.

    Example:
        >>> y_pred = np.asarray([[1,0,0],[1,0,0],[1,1,0]])
        >>> y_true = np.asarray([1,0,1])
        >>> error_overall_ps(y_pred, y_true)
        0.5

    Explanation:
    The 0-th prediction set is {0} and the label is 1, so it is an error (no rejection
    as its prediction set has only one class).
    The 1-th sample is not rejected and incurs on error.
    The 2-th sample is rejected, thus excluded from the computation.
    """
    assert len(y_true.shape) == 1
    truth_pred = y_pred[np.arange(len(y_true)), y_true]
    truth_pred = truth_pred[y_pred.sum(1) == 1]
    return 1 - np.mean(truth_pred)


def _get_bins(bins):
    if isinstance(bins, int):
        bins = list(np.arange(bins+1) / bins)
    return bins

def assign_bin(sorted_ser: pd.Series, bins:int, adaptive:bool=False):
    ret = pd.DataFrame(sorted_ser)
    if adaptive:
        assert isinstance(bins, int)
        step = len(sorted_ser) // bins
        nvals = [step for _ in range(bins)]
        for _ in range(len(sorted_ser) % bins):
            nvals[-_-1] += 1
        ret['bin'] = [ith for ith, val in enumerate(nvals) for _ in range(val)]
        nvals = list(np.asarray(nvals).cumsum())
        bins = [ret.iloc[0]['conf']]
        for iloc in nvals:
            bins.append(ret.iloc[iloc-1]['conf'])
            if iloc != nvals[-1]:
                bins[-1] = 0.5 * bins[-1] + 0.5 *ret.iloc[iloc]['conf']
    else:
        bins = _get_bins(bins)
        bin_assign = pd.Series(0, index=sorted_ser.index)
        locs = [bisect.bisect(sorted_ser.values, b) for b in bins]
        locs[0], locs[-1] = 0, len(ret)
        for i, loc in enumerate(locs[:-1]):
            bin_assign.iloc[loc:locs[i+1]] = i
        ret['bin'] = bin_assign
    return ret['bin'], bins

def _ECE_loss(summ):
    w = summ['cnt'] / summ['cnt'].sum()
    loss = np.average((summ['conf'] - summ['acc']).abs(), weights=w)
    return loss

def _ECE_confidence(df, bins=20, adaptive=False):
    # df should have columns: conf, acc
    df = df.sort_values(['conf']).reset_index().drop('index', axis=1)
    df['bin'], _ = assign_bin(df['conf'], bins, adaptive=adaptive)
    summ = pd.DataFrame(df.groupby('bin')[['acc', 'conf']].mean())#.fillna(0.)
    summ['cnt'] = df.groupby('bin').size()
    summ = summ.reset_index()
    return summ, _ECE_loss(summ)

def _ECE_classwise(prob:np.ndarray, label_onehot:np.ndarray, bins=20, threshold=0., adaptive=False):
    summs = []
    class_losses = {}
    for k in range(prob.shape[1]):
        msk = prob[:, k] >= threshold
        if msk.sum() == 0:
            continue
        df = pd.DataFrame({"conf": prob[msk, k], 'acc': label_onehot[msk, k]})
        df = df.sort_values(['conf']).reset_index()
        df['bin'], _ = assign_bin(df['conf'], bins, adaptive=adaptive)
        summ = pd.DataFrame(df.groupby('bin')[['acc', 'conf']].mean())
        summ['cnt'] = df.groupby('bin').size()
        summ['k'] = k
        summs.append(summ.reset_index())
        class_losses[k] = _ECE_loss(summs[-1])
    class_losses = pd.Series(class_losses)
    class_losses['avg'], class_losses['sum'] = class_losses.mean(), class_losses.sum()
    summs = pd.concat(summs, ignore_index=True)
    return summs, class_losses

def ece_confidence_multiclass(prob:np.ndarray, label:np.ndarray, bins=20, adaptive=False):
    """Expected Calibration Error (ECE).

    We group samples into 'bins' basing on the top-class prediction.
    Then, we compute the absolute difference between the average top-class prediction and
    the frequency of top-class being correct (i.e. accuracy) for each bin.
    ECE is the average (weighed by number of points in each bin) of these absolute differences.
    It could be expressed by the following formula, with :math:`B_m` denoting the m-th bin:

    .. math::
        ECE = \\sum_{m=1}^M \\frac{|B_m|}{N} |acc(B_m) - conf(B_m)|

    Explanation of the example: The bins are [0, 0.5] and (0.5, 1].
    In the first bin, we have one sample with top-class prediction of 0.49, and its
    accuracy is 0. In the second bin, we have average confidence of 0.7 and average
    accuracy of 1. Thus, the ECE is :math:`\\frac{1}{3} \cdot 0.49 + \\frac{2}{3}\cdot 0.3=0.3633`.

    Args:
        prob (np.ndarray): (N, C)
        label (np.ndarray): (N,)
        bins (int, optional): Number of bins. Defaults to 20.
        adaptive (bool, optional): If False, bins are equal width ([0, 0.05, 0.1, ..., 1])
            If True, bin widths are adaptive such that each bin contains the same number
            of points. Defaults to False.
    """
    df = pd.DataFrame({'acc': label == np.argmax(prob, 1), 'conf': prob.max(1)})
    return _ECE_confidence(df, bins, adaptive)[1]

def ece_confidence_binary(prob:np.ndarray, label:np.ndarray, bins=20, adaptive=False):
    """Expected Calibration Error (ECE) for binary classification.

    Similar to :func:`ece_confidence_multiclass`, but on class 1 instead of the top-prediction.


    Args:
        prob (np.ndarray): (N, C)
        label (np.ndarray): (N,)
        bins (int, optional): Number of bins. Defaults to 20.
        adaptive (bool, optional): If False, bins are equal width ([0, 0.05, 0.1, ..., 1])
            If True, bin widths are adaptive such that each bin contains the same number
            of points. Defaults to False.
    """

    df = pd.DataFrame({'acc': label[:,0], 'conf': prob[:,0]})
    return _ECE_confidence(df, bins, adaptive)[1]

def ece_classwise(prob, label, bins=20, threshold=0., adaptive=False):
    """Classwise Expected Calibration Error (ECE).

    This is equivalent to applying :func:`ece_confidence_binary` to each class and take the average.

    Args:
        prob (np.ndarray): (N, C)
        label (np.ndarray): (N,)
        bins (int, optional): Number of bins. Defaults to 20.
        threshold (float): threshold to filter out samples.
            If the number of classes C is very large, many classes receive close to 0
            prediction. Any prediction below threshold is considered noise and ignored.
            In recent papers, this is typically set to a small number (such as 1/C).
        adaptive (bool, optional): If False, bins are equal width ([0, 0.05, 0.1, ..., 1])
            If True, bin widths are adaptive such that each bin contains the same number
            of points. Defaults to False.
    """

    K = prob.shape[1]
    if len(label.shape) == 1:
        # make it one-hot
        label, _ = np.zeros((len(label),K)), label
        label[np.arange(len(label)), _] = 1
    return _ECE_classwise(prob, label, bins, threshold, adaptive)[1]['avg']

def brier_top1(prob:np.ndarray, label:np.ndarray):
    """Brier score (i.e. mean squared error between prediction and 0-1 label) of the top prediction.
    """
    conf = prob.max(1)
    acc: ndarray = (label == np.argmax(prob, 1))
    acc = acc.astype(int)
    return np.mean(np.square(conf - acc))



def binary_metrics_fn(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    metrics: Optional[List[str]] = None,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """Computes metrics for binary classification.

    User can specify which metrics to compute by passing a list of metric names.
    The accepted metric names are:
        - pr_auc: area under the precision-recall curve
        - roc_auc: area under the receiver operating characteristic curve
        - accuracy: accuracy score
        - balanced_accuracy: balanced accuracy score (usually used for imbalanced
          datasets)
        - f1: f1 score
        - precision: precision score
        - recall: recall score
        - cohen_kappa: Cohen's kappa score
        - jaccard: Jaccard similarity coefficient score
        - ECE: Expected Calibration Error (with 20 equal-width bins). Check :func:`pyhealth.metrics.calibration.ece_confidence_binary`.
        - ECE_adapt: adaptive ECE (with 20 equal-size bins). Check :func:`pyhealth.metrics.calibration.ece_confidence_binary`.
    If no metrics are specified, pr_auc, roc_auc and f1 are computed by default.

    This function calls sklearn.metrics functions to compute the metrics. For
    more information on the metrics, please refer to the documentation of the
    corresponding sklearn.metrics functions.

    Args:
        y_true: True target values of shape (n_samples,).
        y_prob: Predicted probabilities of shape (n_samples,).
        metrics: List of metrics to compute. Default is ["pr_auc", "roc_auc", "f1"].
        threshold: Threshold for binary classification. Default is 0.5.

    Returns:
        Dictionary of metrics whose keys are the metric names and values are
            the metric values.

    Examples:
        >>> y_true = np.array([0, 0, 1, 1])
        >>> y_prob = np.array([0.1, 0.4, 0.35, 0.8])
        >>> binary_metrics_fn(y_true, y_prob, metrics=["accuracy"])
        {'accuracy': 0.75}
    """
    if metrics is None:
        metrics = ["pr_auc", "roc_auc", "f1"]

    y_pred = y_prob.copy()
    y_pred[y_pred >= threshold] = 1
    y_pred[y_pred < threshold] = 0

    output = {}
    for metric in metrics:
        if metric == "pr_auc":
            pr_auc = sklearn_metrics.average_precision_score(y_true, y_prob)
            output["pr_auc"] = pr_auc
        elif metric == "roc_auc":
            roc_auc = sklearn_metrics.roc_auc_score(y_true, y_prob)
            output["roc_auc"] = roc_auc
        elif metric == "accuracy":
            accuracy = sklearn_metrics.accuracy_score(y_true, y_pred)
            output["accuracy"] = accuracy
        elif metric == "balanced_accuracy":
            balanced_accuracy = sklearn_metrics.balanced_accuracy_score(y_true, y_pred)
            output["balanced_accuracy"] = balanced_accuracy
        elif metric == "f1":
            f1 = sklearn_metrics.f1_score(y_true, y_pred)
            output["f1"] = f1
        elif metric == "precision":
            precision = sklearn_metrics.precision_score(y_true, y_pred)
            output["precision"] = precision
        elif metric == "recall":
            recall = sklearn_metrics.recall_score(y_true, y_pred)
            output["recall"] = recall
        elif metric == "cohen_kappa":
            cohen_kappa = sklearn_metrics.cohen_kappa_score(y_true, y_pred)
            output["cohen_kappa"] = cohen_kappa
        elif metric == "jaccard":
            jaccard = sklearn_metrics.jaccard_score(y_true, y_pred)
            output["jaccard"] = jaccard
        elif metric in {"ECE", "ECE_adapt"}:
            output[metric] = ece_confidence_binary(
                y_prob, y_true, bins=20, adaptive=metric.endswith("_adapt")
            )
        else:
            raise ValueError(f"Unknown metric for binary classification: {metric}")
    return output


def multiclass_metrics_fn(
        y_true: np.ndarray,
        y_prob: np.ndarray,
        metrics: Optional[List[str]] = None,
        y_predset: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """Computes metrics for multiclass classification.

    User can specify which metrics to compute by passing a list of metric names.
    The accepted metric names are:
        - roc_auc_macro_ovo: area under the receiver operating characteristic curve,
            macro averaged over one-vs-one multiclass classification
        - roc_auc_macro_ovr: area under the receiver operating characteristic curve,
            macro averaged over one-vs-rest multiclass classification
        - roc_auc_weighted_ovo: area under the receiver operating characteristic curve,
            weighted averaged over one-vs-one multiclass classification
        - roc_auc_weighted_ovr: area under the receiver operating characteristic curve,
            weighted averaged over one-vs-rest multiclass classification
        - accuracy: accuracy score
        - balanced_accuracy: balanced accuracy score (usually used for imbalanced
            datasets)
        - f1_micro: f1 score, micro averaged
        - f1_macro: f1 score, macro averaged
        - f1_weighted: f1 score, weighted averaged
        - jaccard_micro: Jaccard similarity coefficient score, micro averaged
        - jaccard_macro: Jaccard similarity coefficient score, macro averaged
        - jaccard_weighted: Jaccard similarity coefficient score, weighted averaged
        - cohen_kappa: Cohen's kappa score
        - brier_top1: brier score between the top prediction and the true label
        - ECE: Expected Calibration Error (with 20 equal-width bins). Check :func:`pyhealth.metrics.calibration.ece_confidence_multiclass`.
        - ECE_adapt: adaptive ECE (with 20 equal-size bins). Check :func:`pyhealth.metrics.calibration.ece_confidence_multiclass`.
        - cwECEt: classwise ECE with threshold=min(0.01,1/K). Check :func:`pyhealth.metrics.calibration.ece_classwise`.
        - cwECEt_adapt: classwise adaptive ECE with threshold=min(0.01,1/K). Check :func:`pyhealth.metrics.calibration.ece_classwise`.

    The following metrics related to the prediction sets are accepted as well, but will be ignored if y_predset is None:
        - rejection_rate: Frequency of rejection, where rejection happens when the prediction set has cardinality other than 1. Check :func:`pyhealth.metrics.prediction_set.rejection_rate`.
        - set_size: Average size of the prediction sets. Check :func:`pyhealth.metrics.prediction_set.size`.
        - miscoverage_ps:  Prob(k not in prediction set). Check :func:`pyhealth.metrics.prediction_set.miscoverage_ps`.
        - miscoverage_mean_ps: The average (across different classes k) of miscoverage_ps.
        - miscoverage_overall_ps: Prob(Y not in prediction set). Check :func:`pyhealth.metrics.prediction_set.miscoverage_overall_ps`.
        - error_ps: Same as miscoverage_ps, but retricted to un-rejected samples. Check :func:`pyhealth.metrics.prediction_set.error_ps`.
        - error_mean_ps: The average (across different classes k) of error_ps.
        - error_overall_ps: Same as miscoverage_overall_ps, but restricted to un-rejected samples. Check :func:`pyhealth.metrics.prediction_set.error_overall_ps`.

    If no metrics are specified, accuracy, f1_macro, and f1_micro are computed
    by default.

    This function calls sklearn.metrics functions to compute the metrics. For
    more information on the metrics, please refer to the documentation of the
    corresponding sklearn.metrics functions.

    Args:
        y_true: True target values of shape (n_samples,).
        y_prob: Predicted probabilities of shape (n_samples, n_classes).
        metrics: List of metrics to compute. Default is ["accuracy", "f1_macro",
            "f1_micro"].

    Returns:
        Dictionary of metrics whose keys are the metric names and values are
            the metric values.

    Examples:
        >>> y_true = np.array([0, 1, 2, 2])
        >>> y_prob = np.array([[0.9,  0.05, 0.05],
        ...                    [0.05, 0.9,  0.05],
        ...                    [0.05, 0.05, 0.9],
        ...                    [0.6,  0.2,  0.2]])
        >>> multiclass_metrics_fn(y_true, y_prob, metrics=["accuracy"])
        {'accuracy': 0.75}
    """
    if metrics is None:
        metrics = ["accuracy", "f1_macro", "f1_micro"]
    prediction_set_metrics = [
        "rejection_rate",
        "set_size",
        "miscoverage_mean_ps",
        "miscoverage_ps",
        "miscoverage_overall_ps",
        "error_mean_ps",
        "error_ps",
        "error_overall_ps",
    ]
    y_pred = np.argmax(y_prob, axis=-1)

    output = {}
    for metric in metrics:
        if metric == "roc_auc_macro_ovo":
            roc_auc_macro_ovo = sklearn_metrics.roc_auc_score(
                y_true, y_prob, average="macro", multi_class="ovo"
            )
            output["roc_auc_macro_ovo"] = roc_auc_macro_ovo
        elif metric == "roc_auc_macro_ovr":
            roc_auc_macro_ovr = sklearn_metrics.roc_auc_score(
                y_true, y_prob, average="macro", multi_class="ovr"
            )
            output["roc_auc_macro_ovr"] = roc_auc_macro_ovr
        elif metric == "roc_auc_weighted_ovo":
            roc_auc_weighted_ovo = sklearn_metrics.roc_auc_score(
                y_true, y_prob, average="weighted", multi_class="ovo"
            )
            output["roc_auc_weighted_ovo"] = roc_auc_weighted_ovo
        elif metric == "roc_auc_weighted_ovr":
            roc_auc_weighted_ovr = sklearn_metrics.roc_auc_score(
                y_true, y_prob, average="weighted", multi_class="ovr"
            )
            output["roc_auc_weighted_ovr"] = roc_auc_weighted_ovr
        elif metric == "accuracy":
            accuracy = sklearn_metrics.accuracy_score(y_true, y_pred)
            output["accuracy"] = accuracy
        elif metric == "balanced_accuracy":
            balanced_accuracy = sklearn_metrics.balanced_accuracy_score(y_true, y_pred)
            output["balanced_accuracy"] = balanced_accuracy
        elif metric == "f1_micro":
            f1_micro = sklearn_metrics.f1_score(y_true, y_pred, average="micro")
            output["f1_micro"] = f1_micro
        elif metric == "f1_macro":
            f1_macro = sklearn_metrics.f1_score(y_true, y_pred, average="macro")
            output["f1_macro"] = f1_macro
        elif metric == "f1_weighted":
            f1_weighted = sklearn_metrics.f1_score(y_true, y_pred, average="weighted")
            output["f1_weighted"] = f1_weighted
        elif metric == "jaccard_micro":
            jacard_micro = sklearn_metrics.jaccard_score(
                y_true, y_pred, average="micro"
            )
            output["jaccard_micro"] = jacard_micro
        elif metric == "jaccard_macro":
            jacard_macro = sklearn_metrics.jaccard_score(
                y_true, y_pred, average="macro"
            )
            output["jaccard_macro"] = jacard_macro
        elif metric == "jaccard_weighted":
            jacard_weighted = sklearn_metrics.jaccard_score(
                y_true, y_pred, average="weighted"
            )
            output["jaccard_weighted"] = jacard_weighted
        elif metric == "cohen_kappa":
            cohen_kappa = sklearn_metrics.cohen_kappa_score(y_true, y_pred)
            output["cohen_kappa"] = cohen_kappa
        elif metric == "brier_top1":
            output[metric] = brier_top1(y_prob, y_true)
        elif metric in {"ECE", "ECE_adapt"}:
            output[metric] = ece_confidence_multiclass(
                y_prob, y_true, bins=20, adaptive=metric.endswith("_adapt")
            )
        elif metric in {"cwECEt", "cwECEt_adapt"}:
            thres = min(0.01, 1.0 / y_prob.shape[1])
            output[metric] = ece_classwise(
                y_prob,
                y_true,
                bins=20,
                adaptive=metric.endswith("_adapt"),
                threshold=thres,
            )
        elif metric in prediction_set_metrics:
            if y_predset is None:
                continue
            if metric == "rejection_rate":
                output[metric] = rejection_rate(y_predset)
            elif metric == "set_size":
                output[metric] = size(y_predset)
            elif metric == "miscoverage_mean_ps":
                output[metric] = miscoverage_ps(y_predset, y_true).mean()
            elif metric == "miscoverage_ps":
                output[metric] = miscoverage_ps(y_predset, y_true)
            elif metric == "miscoverage_overall_ps":
                output[metric] = miscoverage_overall_ps(y_predset, y_true)
            elif metric == "error_mean_ps":
                output[metric] = error_ps(y_predset, y_true).mean()
            elif metric == "error_ps":
                output[metric] = error_ps(y_predset, y_true)
            elif metric == "error_overall_ps":
                output[metric] = error_overall_ps(y_predset, y_true)

        elif metric == "hits@n":
            argsort = np.argsort(-y_prob, axis=1)
            ranking = np.array([np.where(argsort[i] == y_true[i])[0][0] for i in range(len(y_true))]) + 1
            output["HITS@1"] = np.count_nonzero(ranking <= 1) / len(ranking)
            output["HITS@5"] = np.count_nonzero(ranking <= 5) / len(ranking)
            output["HITS@10"] = np.count_nonzero(ranking <= 10) / len(ranking)
        elif metric == "mean_rank":
            argsort = np.argsort(-y_prob, axis=1)
            ranking = np.array([np.where(argsort[i] == y_true[i])[0][0] for i in range(len(y_true))]) + 1
            mean_rank = np.mean(ranking)
            mean_reciprocal_rank = np.mean(1 / ranking)
            output["mean_rank"] = mean_rank
            output["mean_reciprocal_rank"] = mean_reciprocal_rank

        else:
            raise ValueError(f"Unknown metric for multiclass classification: {metric}")

    return output

