from __future__ import division

import numpy as np

from .detection_base import DetectionBaseScoreType
from .precision_recall import _match_tuples


def score_craters_on_patch(y_true, y_pred, cut_off=1, minipatch=None):
    """
    Main OSPA score for single patch.

    Parameters
    ----------
    y_true : list of tuples (x, y, radius)
        List of coordinates and radius of actual craters in a patch
    y_pred : list of tuples (x, y, radius)
        List of coordinates and radius of craters predicted in the patch

    Returns
    -------
    float : score for a given path, the lower the better

    """
    y_true = np.atleast_2d(y_true).T
    y_pred = np.atleast_2d(y_pred).T
    score = ospa_single(y_true, y_pred, cut_off=cut_off, minipatch=minipatch)
    return score


def ospa_single(y_true, y_pred, cut_off=1, minipatch=None):
    """
    OSPA score on single patch. See docstring of `ospa` for more info.

    Parameters
    ----------
    y_true, y_pred : ndarray of shape (3, x)
        arrays of (x, y, radius)
    cut_off : float, optional (default is 1)
        penalizing value for wrong cardinality
    minipatch : list of int, optional
        Bounds of the internal scoring patch (default is None)

    Returns
    -------
    float: distance between input arrays

    """
    x_size = y_true.size
    y_size = y_pred.size

    _, m = y_true.shape
    _, n = y_pred.shape

    if m > n:
        return ospa_single(y_pred, y_true, cut_off, minipatch)

    # NO CRATERS
    # ----------
    # GOOD MATCH
    if x_size == 0 and y_size == 0:
        return 0

    # BAD MATCH
    if x_size == 0 or y_size == 0:
        return cut_off

    # minipatch cuts
    if minipatch is not None:
        row_min, row_max, col_min, col_max = minipatch

        y_true_cut = ((y_true[0] >= col_min) & (y_true[0] < col_max) &
                      (y_true[1] >= row_min) & (y_true[1] < row_max))
        y_pred_cut = ((y_pred[0] >= col_min) & (y_pred[0] < col_max) &
                      (y_pred[1] >= row_min) & (y_pred[1] < row_max))

        y_true = y_true[y_true_cut]
        y_pred = y_pred[y_pred_cut]

    # OSPA METRIC
    _, _, ious = _match_tuples(y_true.T.tolist(), y_pred.T.tolist())
    iou_score = ious.sum()

    distance_score = m - iou_score
    cardinality_score = cut_off * (n - m)

    dist = 1 / n * (distance_score + cardinality_score)

    return dist


class OSPA(DetectionBaseScoreType):
    is_lower_the_better = True
    minimum = 0.0
    maximum = 1.0

    def __init__(self, name='ospa', precision=2, conf_threshold=0.5,
                 cut_off=1, minipatch=None):
        self.name = name
        self.precision = precision
        self.conf_threshold = conf_threshold
        self.minipatch = minipatch
        self.cut_off = cut_off

    def detection_score(self, y_true, y_pred):
        """Optimal Subpattern Assignment (OSPA) metric for IoU score.

        This metric provides a coherent way to compute the miss-distance
        between the detection and alignment of objects. Among all
        combinations of true/predicted pairs, if finds the best alignment
        to minimise the distance, and still takes into account missing
        or in-excess predicted values through a cardinality score.

        The lower the value the smaller the distance.

        Parameters
        ----------
        y_true, y_pred : list of list of tuples

        Returns
        -------
        float: distance between input arrays

        References
        ----------
        http://www.dominic.schuhmacher.name/papers/ospa.pdf

        """
        scores = [score_craters_on_patch(t, p, self.cut_off, self.minipatch)
                  for t, p in zip(y_true, y_pred)]
        weights = [len(t) for t in y_true]
        return np.average(scores, weights=weights)
