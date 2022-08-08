import torch
from torch import nn
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

import argparse
import math
import os
import random
import time


def denormalize(x, x_min, x_max):
    # Denormalize ground truth and predicted error

    if torch.is_tensor(x):
        ret = ((x + 1) * (x_max - x_min)) / 2 + x_min
    elif len(x):
        ret = [((xi + 1) * (x_max - x_min)) / 2 + x_min for xi in x]
    else:
        x = float(x)
        ret = ((x + 1) * (x_max - x_min)) / 2 + x_min

    return ret



def calc_precision(gt, pred, threshold):

    errors = []
    neg_errors = []
    gt = list(gt)
    pred = list(pred)

    correct = 0 # for precision calculation
    negsign_correct = 0 # for precision calculation if gt is negative
    gt_negatives = 0 # negative signed ground truth
    pred_negatives = 0   # negative signed predictions

    for g, p in zip(gt, pred):

        err = abs(denormalize(g, 0, 1) - denormalize(p, 0 ,1))
        if err < 0.01: correct += 1
        errors.append(err)

        if g < 0:
            gt_negatives += 1
            neg_errors.append(err)
            if p < 0: pred_negatives += 1
            if err < threshold: negsign_correct += 1

        #print(" gt ", denormalize(g, 0, 1), " pred: ",denormalize(p, 0, 1),  )
        #avg_gt = sum(gt)/len(gt)
        #print(avg_gt)
        #if (avg_gt-g>0 and avg_gt-p>0) or (avg_gt-g<0 and avg_gt-p<0) or (avg_gt-g==0 and avg_gt-p==0): sign_correct += 1

    error = sum(errors)/len(errors)
    neg_error = sum(neg_errors)/(len(neg_errors) + 1e-15)
    precision = correct/len(gt)
    negsign_hits = pred_negatives/(gt_negatives + 1e-15)
    negsign_precision = negsign_correct/(gt_negatives + 1e-15)

    #print("batch avg error ", error)
    #print("negerr ", negative_errors, " negcorr ", negative_corrects)
    #sign_precision = sign_correct/len(gt)

    results = np.array([error.item(), precision, neg_error.item(), negsign_hits, negsign_precision])

    return results