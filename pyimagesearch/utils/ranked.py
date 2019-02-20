__author__ = 'kevin'

import numpy as np


def rank5_accuracy(preds, lables):
    rank1 = 0
    rank5 = 0

    for (p, gt) in zip(preds, lables):
        p = np.argsort(p)[::-1]

        if gt in p[:5]:
            rank5 += 1

        if gt == p[0]:
            rank1 += 1

    rank1 /= float(len(lables))
    rank5 /= float(len(lables))

    return (rank1, rank5)
