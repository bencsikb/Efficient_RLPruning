import torch
from torch import nn
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

import argparse
import math
import os
import random
import sys

from prune_for_error import *
from test import *

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


def normalize(x, x_min, x_max):
    # Between -1 and 1
    if torch.is_tensor(x):
        x = torch.FloatTensor(x)
    else:
        x = float(x)
    x = 2 * ((x - x_min) / (x_max - x_min)) - 1
    return x


def get_state(prev_state, action_seq, network_seq, layer, layer_index, layer_cnt, init_param_nmb, yolo_layers,
              timefile):
    """
    layer_inxdex: layer's index in the network
    layer_cnt: index of layer that can be pruned

    state[0]: input_channels
    state[1]: output_channels
    state[2]: kernel_size
    state[3]: stride
    state[4]: padding
    state[5]: percent of pruned parameters from current layer
    state[6]: percent of pruned parameters form the whole network
    """

    state = torch.full(prev_state.shape[:2], -1.0)
    new_network_seq = network_seq

    # print("Networks: ")
    # for n in network_seq:
    #    print( sum([param.nelement() for param in n.parameters()]))

    for i, (alpha, network) in enumerate(zip(action_seq[:, 0, layer_cnt], network_seq)):

        # network = network_seq[i]
        network_param_nmb = sum([param.nelement() for param in network.parameters()])
        layer_param_nmb_before = sum(
            [param.nelement() for name, param in network.named_parameters() if "." + str(layer_index) + "." in name])

        state[i, 0] = normalize(layer.in_channels, 0, 1024)
        state[i, 1] = normalize(layer.out_channels, 0, 1024)
        state[i, 2] = normalize(layer.kernel_size[0], 0, 3)
        state[i, 3] = normalize(layer.stride[0], 0, 2)
        state[i, 4] = normalize(layer.padding[0], 0, 1)

        if denormalize(alpha, 0.0, 2.2) != 0.0:

            start_time = time.time()
            network = prune_network(network, yolo_layers, layer_index, denormalize(alpha, 0.0, 2.2), device='cuda')
            # print("Pruning time/batch ", time.time() - start_time)
            # with open(timefile, "a+") as f:
            # f.write("Pruning time/batch " + str(time.time() - start_time) + "\n")

            param_nmb_after = sum([param.nelement() for param in network.parameters()])
            layer_param_nmb_after = sum([param.nelement() for name, param in network.named_parameters() if
                                         "." + str(layer_index) + "." in name])

            state[i, 5] = normalize(1.0 - float(layer_param_nmb_after) / float(layer_param_nmb_before), 0, 1)
            state[i, 6] = normalize(float(1 - param_nmb_after / init_param_nmb), 0, 1)

            # print(layer_param_nmb_before, layer_param_nmb_after)


        else:

            # print("Alpha = 0")
            state[i, 5] = normalize(layer.padding[0], 0, 1)
            # state[i, 6] = pruned_params_prev[i]
            state[i, 6] = prev_state[i, 6, layer_cnt - 1]

        new_network_seq[i] = network

    # print("New networks: ")
    # for n in new_network_seq:
    #    print(sum([param.nelement() for param in n.parameters()]))

    return state, new_network_seq


def get_state2(state, sparsity, layer, layer_cnt):
    i = layer_cnt

    state[:, 0, i] = normalize(layer.in_channels, 0, 1024)
    state[:, 1, i] = normalize(layer.out_channels, 0, 1024)
    state[:, 2, i] = normalize(layer.kernel_size[0], 0, 3)
    state[:, 3, i] = normalize(layer.stride[0], 0, 2)
    state[:, 4, i] = normalize(layer.padding[0], 0, 1)
    state[:, 5, i] = sparsity

    return state


def get_layers_forpruning(network, yolo_layers):
    layers_to_prune = []

    dim = 0  # 0: prune corresponding dim of filter weight [out_ch, in_ch, k1, k2]
    network_size = len(network.module_list)
    dims = [0] * 1

    for i in range(network_size):

        sequential_size = len(network.module_list[i])
        module_def = network.module_defs[i]

        if module_def["type"] == "route":
            dims.append(1)
        elif module_def["type"] == "shortcut":
            dims.append(1)
        elif module_def["type"] == "yolo":
            dims.append(dims[-1])
        elif module_def["type"] == "maxpool" or module_def["type"] == "upsample":
            dims[-1] = 0
            dims.append(0)
        elif module_def["type"] == "convolutional":

            for j in range(sequential_size):

                layer = network.module_list[i][j]

                if isinstance(layer, nn.Conv2d):

                    if (i in yolo_layers) and (dim == 0):
                        dims.append(0)  # dont care
                        break

                    if len(dims):
                        dim = dims[-1]
                    else:
                        dim = 0

                    if (dim == 0):  # get new alpha
                        layers_to_prune.append(i)

                    dim ^= 1
                    dims.append(dim)

    return layers_to_prune


def list2FloatTensor(lisst):
    # shape = [len(lisst)] + list(lisst[0].size())
    torch_tens = torch.zeros((len(lisst), lisst[0].shape[0], lisst[0].shape[1]))
    # torch_tens = torch.zeros(shape)

    for i in range(len(lisst)):
        torch_tens[i, :, :] = lisst[i]

    return torch_tens

    """

    torch_tens = torch.FloatTensor(lisst[0].cpu())
    print("first", torch_tens)

    for i in range(len(lisst) - 1):
        #torch_tens = [torch_tens, torch.FloatTensor(lisst[i + 1].cpu())]
        torch.stack([torch_tens, torch.FloatTensor(lisst[i + 1].cpu())], out=torch_tens)

        print("final", torch_tens)
    return torch_tens

    """

def test_alpha_seq(error, sparsity, reward, alpha_seq, yolo_layers, test_case, error_thresh=None, spars_thresh=None, reward_thresh=None, save_flag=False):

    """
    :param error [batch_size, 1]: error for each sample in the batch after pruning the last layer
    :param sparsity [batch_size, 1]: spars. for each sample in the batch after pruning the last layer
    :param reward [1, batch_size]: reward for each sample in the batch after pruning the last layer
    :param action_seq [batch_size, 44]:
    :param error_thresh:
    :param spars_thresh:
    :param reward_thresh:
    :return:
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default="weights/yolov4_kitti.weights",
                        help='model.pt path(s)')
    parser.add_argument('--data', type=str, default='data/kitti.yaml', help='*.data path')
    parser.add_argument('--batch-size', type=int, default=1, help='size of each image batch')
    parser.add_argument('--img-size', type=int, default=540, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.001, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.5, help='IOU threshold for NMS')
    parser.add_argument('--save-json', action='store_true', help='save a cocoapi-compatible JSON results file')
    parser.add_argument('--task', default='val', help="'val', 'test', 'study'")
    parser.add_argument('--device', default='cuda', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--single-cls', action='store_true', help='treat as single-class dataset')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--merge', action='store_true', help='use Merge NMS')
    parser.add_argument('--verbose', action='store_true', help='report mAP by class')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--cfg', type=str, default='cfg/yolov4_kitti.cfg', help='*.cfg path')
    parser.add_argument('--names', type=str, default='data/kitti.names', help='*.cfg path')
    parser.add_argument('--save-path', type=str, default='sandbox/DatasetForPruning')
    parser.add_argument('--test-cases', type=int, default=5000)
    parser.add_argument('--map_before', type=int, default=0.6850150267)

    opt = parser.parse_args()

    batch_size = error.shape[0]
    error = error.squeeze(0)
    sparsity = sparsity.squeeze(0)
    reward = reward.squeeze(0)
    layer_to_prune = 43
    save_path = F"final_results/{test_case}.txt"
    save_excel_path = F"final_results/exc_{test_case}.txt"


    for bi in range(batch_size):

        with open(save_path, 'a+') as f:
            try:

                examine_flag = False

                if error_thresh is not None and error[bi] < error_thresh:
                    examine_flag = True
                    print("Error falls below threshold.")
                    f.write("\nError falls below threshold.\n")
                elif spars_thresh is not None and sparsity[bi] > sparsity_thresh:
                    examine_flag = True
                    print("Sparsity is above threshold.")
                    f.write("\nSparsity is above threshold.\n")
                if reward is not None and reward[bi] > reward_thresh:
                    examine_flag = True
                    #print("Reward is above threshold.")
                    f.write("\nReward is above threshold.\n")

                if examine_flag and bi>580:
                #if bi == 23:

                    print(F"Sample: {bi}, Error: {error[bi]}, Sparsity: {sparsity[bi]}, Reward: {reward[bi]}")
                    print(alpha_seq[bi, :])
                    f.write(F"Sample: {bi}, Error: {error[bi]}, Sparsity: {sparsity[bi]}, Reward: {reward[bi]}\n")
                    f.write(str(alpha_seq[bi, :]) + "\n")

                    # Load trained YOLOv4 net
                    net_for_pruning = Darknet(opt.cfg).to(opt.device)
                    net_for_pruning.load_darknet_weights(opt.weights)
                    #ckpt_nfp = torch.load(opt.network_forpruning)
                    #state_dict = {k: v for k, v in ckpt_nfp['model'].items() if
                    #              net_for_pruning.state_dict()[k].numel() == v.numel()}
                    #net_for_pruning.load_state_dict(state_dict, strict=False)

                    # Pruning
                    #alpha_seq = torch.tensor(
                    #    [0.0, 0., 0., 0., 0., 0., 0, 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0, 0., 0., 0., 0., 0., 0., 0.,
                    #     0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0, 0., 0., 0., 0., 0.])
                    param_nmb_before = sum([param.nelement() for param in net_for_pruning.parameters()])
                    print("param_nmb_before", param_nmb_before)
                    pruned_network = prune_network(net_for_pruning, yolo_layers, layer_to_prune, alpha_seq[bi,:], device='cuda')
                    param_nmb_after = sum([param.nelement() for param in pruned_network.parameters()])
                    print("param_nmb_after", param_nmb_after)
                    f.write(F"param before: {param_nmb_before}, param after: {param_nmb_after}, ")

                    # Real sparsity
                    real_sparsity = float(1 - param_nmb_after / param_nmb_before)

                    # Real error
                    results, _, _ = test(
                        opt.data,
                        batch_size=1,
                        imgsz=540,
                        conf_thres=0.001,
                        iou_thres=0.5,
                        save_json=False,  # save json
                        single_cls=opt.single_cls,
                        augment=opt.augment,
                        verbose=opt.verbose,
                        model=pruned_network.to(opt.device),
                        opt=opt,
                        called_directly=True)

                    prec_after, rec_after, map_after = results[0], results[1], results[2]
                    print("map_after ", map_after)
                    f.write(F"mAP after: {map_after}\n")
                    real_error = 1.0 - (float(map_after) / float(opt.map_before))

                    print(F"Real error: {real_error}, Real_spars: {real_sparsity}")
                    f.write(F"Real error: {real_error}, Real_spars: {real_sparsity}\n")

                    with open(save_excel_path, 'a+') as ef:
                        ef.write(F"{bi} {real_error} {real_sparsity}\n")


            except KeyboardInterrupt:
                print("KeyboardInterrupt")
                sys.exit(0)
            except:
                f.write(F"Sample: {bi}, Exception catced\n")
                print("catched")
                continue

