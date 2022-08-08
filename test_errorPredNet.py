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

from utils.errorPredNet_utils import denormalize, calc_precision


def validate(dataloader, model, criterion_err, criterion_spars):

    metrics_sum_err = np.zeros((1, 5))
    metrics_sum_spars = np.zeros((1, 5))

    running_loss = 0
    cnnt = 0

    for batch_i, (data, label_gt) in enumerate(dataloader):
        data = data.type(torch.float32).cuda()
        data = torch.cat((data[:, :44], data[:, 264:]), dim=1)
        label_gt = label_gt.type(torch.float32).cuda()
        prediction = model(data)
        loss = criterion_err(denormalize(label_gt[:, 0], 0, 1), denormalize(prediction[:, 0], 0, 1)) + criterion_spars(denormalize(label_gt[:, 1], 0, 1), denormalize(prediction[:, 1], 0, 1))
        running_loss += loss.cpu().item()
        metrics_sum_err += calc_precision(label_gt[:, 0], prediction[:, 0], threshold=0.01)
        metrics_sum_spars += calc_precision(label_gt[:, 1], prediction[:, 1], threshold=0.01)


    print("valid cnt ", cnnt)

    # Calculate validation metrics

    running_loss /= len(dataloader)
    metrics_avg_err = metrics_sum_err / len(dataloader)
    metrics_avg_spars = metrics_sum_spars / len(dataloader)


    return running_loss, metrics_avg_err, metrics_avg_spars
