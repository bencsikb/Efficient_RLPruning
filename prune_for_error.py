import torch
import argparse
import torch.nn as nn
import math
from numpy import linalg as LA
import random
import numpy as np
import itertools
import time

from models.models import *
from utils.layers import *
from test import *
from utils.datasets import *
from utils.general import *

global_print = False
global_filewrite = False


def choose_channels_to_prune(layer, alpha, dim):
    indexes = []
    norms_list = []

    start_time = time.time()

    # Count norm for every channel

    if dim == 0:
        norms = (torch.norm(layer.weight.data, 'fro', dim=[2, 3]))
        print("Layer dimenzions: ", layer.weight.data.shape[2], layer.weight.data.shape[3] )
        norms = torch.norm(norms, 'fro', dim=1)
        # print(norms.device)

        """ This solution works fine, uncomment this for testing it. """
        # for i in range(layer.out_channels):
        # norms_list.append(LA.norm(layer.weight.data[i, :, :, :].cpu()))
        # print(norms_list)

    else:  # dim == 1
        norms = (torch.norm(layer.weight.data, 'fro', dim=[2, 3]))
        norms = torch.norm(norms, 'fro', dim=0)

    #print("Norm calculations ", time.time() - start_time)

    # Count the average and standard deviation of the channel norms in the layer
    norms_avg = torch.mean(norms)
    norms_std = torch.std(norms)

    if global_print: print("avg ", norms_avg)
    if global_print: print("std ", norms_std)

    # Find the indexes of the channels that have to be pruned
    start_time = time.time()

    #indexes = [i for i, norm in enumerate(norms) if ( (torch.absolute(norm) < norms_avg + alpha * norms_std) and (torch.absolute(norm) > norms_avg - alpha * norms_std)) ]

    norm_condition = torch.logical_and((torch.absolute(norms) < norms_avg + alpha * norms_std), (torch.absolute(norms) > norms_avg - alpha * norms_std))
    indexes = norm_condition.nonzero()

    #print("indexes ", indexes[:20], " indexes2 ",  indexes2[:20])

    #print("choosing liscomp ", time.time() - start_time)
    start_time = time.time()

    #for i, norm in enumerate(norms):
        # ! if (np.absolute(norm) < norms_avg + alpha * norms_std) and (np.absolute(norm) > norms_avg - alpha * norms_std):
        #if (torch.absolute(norm) < norms_avg + alpha * norms_std) and (torch.absolute(norm) > norms_avg - alpha * norms_std):
            #print("smaller norm ", i, " ", norm)
            #continue
            #indexes.append(i)

    #print("choosing for loop ", time.time() - start_time)

    # Avoid pruning the whole layer
    if (dim == 0) and (len(indexes) == layer.out_channels):
        c = random.choice(np.arange(0, layer.out_channels - 1, 1).tolist())
        print("avoiding", c)
        #indexes.pop(c)
        indexes = indexes[1:]

    if global_print: print("len(indexes) ", len(indexes))
    return indexes


def get_route_indexes(dims, routes, index_container, old_output_filters, f, device):
    indexes = torch.empty((0,1)).to(device)
    adding = 0

    if global_filewrite: f.write("old outputs " + str(old_output_filters) + "\n")
    # with open(file, 'a+') as f:
    if True:
        if global_filewrite: f.write("********** Get route indexes *********\n")
        for k in routes:
            if global_filewrite: f.write("k " + str(k) + "\n")
            if global_print: print("k ", k)
            # print(len(index_container[k]))
            # k = k-1
            if global_filewrite: f.write("dims " + str(dims) + "\n")
            if index_container[k] is not None:
                if global_filewrite: f.write("len(index_container[k]) " + str(len(index_container[k])) + "\n")
            else:
                if global_filewrite:  f.write("index cont in None \n")

            if global_print: print(dims)
            if global_print: print(dims[-1])
            if dims[k - 1] == 0:
                if global_filewrite: f.write("HERE\n")
                if global_print: print("HERE")

                #print(type(index_container[k]))
                #print(len(index_container[k]))

                #indexes += list(np.asarray(index_container[k]) + adding)
                indexes = torch.cat((indexes, index_container[k] + adding), dim=0)
                #if (len(index_container[k])):
                #    print(len(index_container[k]))
                #    print("catresult ", indexes.shape)

                if global_filewrite: f.write("adding " + str(adding) + "\n")
                # adding = adding + len(index_container[k])
                adding = adding + old_output_filters[k]

        if global_filewrite: f.write(str(indexes) + "\n")
        if len(indexes):
            if global_filewrite: f.write("len(indexes) " + str(len(indexes)) + "\n")
        else:
            if global_filewrite: f.write(" len(indexes) 0 \n")
        if global_filewrite:  f.write("********** ***************\n")
    return indexes


def get_shortcut_indexes(dims, fromm, index_container, output_filters):
    min_channel = output_filters[-1]
    indexes = index_container[-1]

    if output_filters[fromm] < min_channel:
        min_channel = output_filters[fromm]
        indexes = index_container[fromm]

    input_channels = min_channel

    return indexes


def prune_network(network, yolo_layers, layer_to_prune, alpha_seq, device='cpu', dataset_make=False):
    # def prune_network(network, yolo_layers):
    # ! network = network.cpu()
    print("In prune network finc")

    # if dim == 0 --> prune output channels
    # if dim == 1 --> prune input channels
    dim = 0  # 0: prune corresponding dim of filter weight [out_ch, in_ch, k1, k2]
    layer_cnt = 0
    network_size = len(network.module_list)
    dims = [0] * 1
    alphas = []
    prev_indexes = []
    output_filters = []
    old_output_filters = []
    feature_concat_cont = []
    feature_fusion_cont = []
    index_container = []
    index_lens = []
    yolo_flag = False

    parser = {'dim': dim,
              'in_ch': 0,
              'out_ch': 0,
              'kernel': 0,
              'stride': 0,
              'pad': 0,
              'nmb_of_pruned_ch': 0}

    save_path = "/home/blanka/YOLOv4_Pruning/sandbox/" + "pruning_develop_data.txt"
    with open(save_path, 'w') as f:
        # if True:
        time_sum = 0
        for i in range(network_size):

            start_time_forloop = time.time()

            sequential_size = len(network.module_list[i])
            module_def = network.module_defs[i]
            if global_filewrite: f.write("\nmodule_def " + str(module_def["type"]) + str(module_def) + "\n")

            if module_def["type"] == "route":
                routes = [int(x) for x in module_def["layers"]]
                filters = sum([output_filters[l + 1 if l > 0 else l] for l in routes])
                old_filters = sum([old_output_filters[l + 1 if l > 0 else l] for l in routes])
                indexes = get_route_indexes(dims, routes, index_container, old_output_filters, f, device)
                alpha = alphas[routes[-1]]
                if global_filewrite: f.write("route dims" + str(dims) + "\n")
                # f.write("route alphas"+str(alphas)+ "\n")
                if global_filewrite: f.write("route output_filters" + str(output_filters) + "\n")
                if global_filewrite: f.write("route index lens " + str(index_lens) + "\n")
                dims.append(1)
                output_filters.append(filters)
                old_output_filters.append(old_filters)
                index_container.append(indexes)
                index_lens.append(len(indexes))
            elif module_def["type"] == "shortcut":
                fromm = int(module_def["from"][0])
                filters = output_filters[1:][int(module_def["from"][0])]
                old_filters = old_output_filters[1:][int(module_def["from"][0])]
                indexes = get_shortcut_indexes(dims, fromm, index_container, output_filters)
                alpha = alphas[fromm]
                if global_filewrite: f.write("shortcut dims" + str(dims) + "\n")
                # f.write("shortcut alphas"+str(alphas)+ "\n")
                if global_filewrite: f.write("shortcut output_filters" + str(output_filters) + "\n")
                if global_filewrite: f.write("shortcut index lens " + str(index_lens) + "\n")
                dims.append(1)
                output_filters.append(filters)
                old_output_filters.append(old_filters)
                index_container.append(indexes)
                if indexes is not None:
                    index_lens.append(len(indexes))
                else:
                    index_lens.append(0)
            elif module_def["type"] == "yolo":
                alpha = alphas[-1]
                if global_filewrite: f.write("yolo dims" + str(dims) + "\n")
                # f.write("yolo alphas"+str(alphas)+ "\n")
                if global_filewrite: f.write("yolo output_filters" + str(output_filters) + "\n")
                dims.append(dims[-1])
                output_filters.append(0)
                old_output_filters.append(0)
                index_container.append([])
                index_lens.append(len(indexes))

            elif module_def["type"] == "maxpool" or module_def["type"] == "upsample":
                alpha = alphas[-1]
                if global_filewrite: f.write("maxpool dims" + str(dims) + "\n")
                # f.write("maxpool alphas"+str(alphas)+ "\n")
                if global_filewrite: f.write("maxpool output_filters" + str(output_filters) + "\n")
                dims[-1] = 0
                dims.append(0)
                output_filters.append(output_filters[-1])
                old_output_filters.append(old_output_filters[-1])
                index_container.append(index_container[-1])
                index_lens.append(len(indexes))
            elif module_def["type"] == "convolutional":

                for j in range(sequential_size):

                    layer = network.module_list[i][j]

                    if global_filewrite: f.write("Pruning layer " + str(i) + " " + str(layer) + "\n")
                    # print("File writing ", start_time - time.time())
                    # start_time = time.time()

                    if isinstance(layer, nn.Conv2d):

                        old_output_filters.append(layer.weight.shape[0])
                        if (i in yolo_layers) and (dim == 0):
                            # if i in yolo_layers:
                            output_filters.append(255)
                            dims.append(0)  # dont care
                            if global_filewrite: f.write("breaked at layer " + str(i))
                            break
                        elif (i in yolo_layers) and (dim == 1):
                            yolo_flag = True

                        if len(dims):
                            dim = dims[-1]
                        else:
                            dim = 0

                        if (dim == 0):  # get new alpha

                            #print(alpha_seq[layer_cnt])
                            alpha = float(torch.round(alpha_seq[layer_cnt]).item())
                            layer_cnt += 1
                            indexes = choose_channels_to_prune(layer, alpha, dim)

                            index_container.append(indexes)
                            index_lens.append(len(indexes))

                        else:  # dim == 1

                            if global_filewrite: f.write("else branch \n")
                            if global_filewrite: f.write(str(prev_indexes))
                            indexes = index_container[-1]
                            index_container.append(torch.empty((0,1)).to(device))
                            index_lens.append(0)

                        # Logging

                        if i == layer_to_prune:
                            parser['dim'] = dim
                            parser['in_ch'] = layer.in_channels
                            parser['out_ch'] = layer.out_channels
                            parser['kernel'] = layer.kernel_size[0]
                            parser['stride'] = layer.stride[0]
                            parser['pad'] = layer.padding[0]
                            parser['nmb_of_pruned_ch'] = len(indexes)
                            # parser['channels_before'] = layer.in_channels if dim==1 else layer.out_channels

                        if global_filewrite: f.write("conv dims " + str(dims) + "\n")
                        # f.write("conv alphas"+str(alphas)+ "\n")
                        if global_filewrite: f.write("conv output_filters " + str(output_filters) + "\n")
                        if global_filewrite:  f.write("conv dim " + str(dim) + "\n")
                        if indexes is not None:
                            if global_filewrite: f.write("conv indexes " + str(len(indexes)) + "\n")
                        else:
                            if global_filewrite: f.write("conv indexes " + str(0) + "\n")
                        if global_filewrite: f.write("conv index lens " + str(index_lens) + "\n")

                        if len(indexes) and alpha != 0:
                            # if indexes is not None:

                            if global_print: print("alpha ", alpha, "indexes ", len(indexes))
                            start_time = time.time()
                            new_conv, new_def = get_new_conv(layer, indexes, dim, module_def, yolo_flag)
                            #print("Get new conv  ", time.time() -start_time)
                            network.module_list[i][j] = new_conv
                            network.module_defs[i] = new_def

                            if global_filewrite: f.write("New conv layer: " + str(new_conv) + "\n")
                            if global_filewrite: f.write("New conv def: " + str(new_def) + "\n")

                        dim ^= 1
                        dims.append(dim)
                        if global_print: print(dims)
                        yolo_flag = False
                        output_filters.append(network.module_defs[i]["filters"])


                    elif isinstance(layer, nn.BatchNorm2d) and (dim == 1):

                        if global_filewrite: f.write("bnorm dims" + str(dims) + "\n")
                        if global_filewrite: f.write("bnorm alphas" + str(alphas) + "\n")

                        if global_filewrite: f.write("bnorm dim" + str(dim) + "\n")
                        if global_filewrite: f.write("bnorm indexes" + str(indexes) + "\n")

                        if len(indexes):
                            # if indexes is not None:
                            new_norm = get_new_norm(layer, indexes, dim)
                            network.module_list[i][j] = new_norm

                            if global_filewrite: f.write("New cbn layer: " + str(new_norm) + "\n")
            # dims.append(dim)
            alphas.append(alpha)

            #if time.time() - start_time_forloop > 0.0002:
            #    print("For loop in prune_netwok ", i, time.time() - start_time_forloop)
            #    time_sum += time.time() - start_time_forloop
            #if i == 160: print("timesum ", time_sum)

    if dataset_make:
        return network, parser

    else:
        return network


def get_new_conv(layer, indexes, dim, module_def, yolo_flag):
    if global_print: print("yolo_flag ", yolo_flag)
    if yolo_flag:
        new_bias = False
    else:
        new_bias = layer.bias

    """
    if yolo_flag:
        new_bias = True
    else:
        new_bias = layer.bias
    """

    if dim == 0:

        out_planes = int(layer.out_channels) - len(indexes)

        new_conv = nn.Conv2d(in_channels=layer.in_channels,
                             out_channels=out_planes,
                             kernel_size=layer.kernel_size,
                             stride=layer.stride,
                             padding=layer.padding,
                             bias=new_bias)

        module_def['filters'] = out_planes

        new_conv.weight.data = get_weights(layer.weight.data, indexes, dim)
        if global_print: print("New layer ", new_conv.weight.data.shape)
        """
        if layer.bias is not None:
            if not yolo_flag:
                new_conv.bias.data = get_weights(layer.bias.data, indexes, dim, onedim=True)
            else:
                new_conv.bias.data = layer.bias.data
        """

        if new_bias:
            new_conv.bias.data = get_weights(layer.bias.data, indexes, dim, onedim=True)


    else:
        in_planes = int(layer.in_channels) - len(indexes)

        new_conv = nn.Conv2d(in_channels=in_planes,
                             out_channels=layer.out_channels,
                             kernel_size=layer.kernel_size,
                             stride=layer.stride,
                             padding=layer.padding,
                             bias=new_bias)

        new_conv.weight.data = get_weights(layer.weight.data, indexes, dim)
        if global_print: print("New layer ", new_conv.weight.data.shape)
        # if layer.bias is not None:
        if new_bias:
            new_conv.bias.data = get_weights(layer.bias.data, indexes, dim, onedim=True)

        if yolo_flag:
            new_conv.bias = layer.bias
            new_conv.bias.data = layer.bias.data

    return new_conv, module_def


def get_weights(weights, indexes, dim, onedim=False, running_stat=False):
    if not onedim:
        if global_print: print("weights.size(dim) ", weights.size(dim))
        if global_print: print(weights.shape)
        size = list(weights.size())
        new_size = size
        new_size[dim] = weights.size(dim) - len(indexes)
        if global_print: print(new_size)

        new_weights = torch.zeros(new_size).cuda()
        if global_print: print(new_weights.shape)

        cnt = 0
        for i in range(weights.shape[dim]):
            if i not in indexes:

                if dim == 0:
                    new_weights[cnt, ...] = weights[i, ...]
                else:
                    if weights.shape[2] == 1:
                        new_weights[:, cnt] = weights[:, i]
                    else:
                        # new_weights[:, cnt, :, :] = weights[:, i, :,:]
                        new_weights[:, cnt, ...] = weights[:, i, ...]

                cnt += 1

        # print("old weights ",weights[5,0])
        # print("new weights ", new_weights[5,0])

    elif (onedim and not running_stat) or (running_stat and dim == 1):

        new_size = list(weights.size())[0] - len(indexes)
        new_weights = torch.zeros(new_size)
        # print("onedim weights shape ", weights.shape, weights)

        cnt = 0
        # for i in range(weights.shape[0]):
        for i in range(new_size):
            if i not in indexes:
                new_weights[cnt] = weights[i]
                cnt += 1

        # print("old BIAS ",weights)
        # print("new BIAS ", new_weights)

    else:

        new_weights = weights

    return new_weights


def get_new_norm(layer, indexes, dim):
    new_norm = nn.BatchNorm2d(num_features=int(layer.num_features - len(indexes)),
                              eps=layer.eps,
                              momentum=layer.momentum,
                              affine=layer.affine,
                              track_running_stats=layer.track_running_stats)

    new_norm.weight.data = get_weights(layer.weight.data, indexes, 0, onedim=True)
    if global_print: print("Getting new norm weights layer. Shape: ", new_norm.weight.data.shape)
    if layer.bias is not None:
        new_norm.bias.data = get_weights(layer.bias.data, indexes, 0, onedim=True)
        if global_print: print("Getting new norm bias layer. Shape: ", new_norm.bias.data.shape)

    if layer.track_running_stats is not None:
        new_norm.running_mean.data = get_weights(layer.running_mean.data, indexes, dim, onedim=True, running_stat=True)
        if global_print: print("Getting new norm running mean layer. Shape: ", new_norm.running_mean.data.shape)
        new_norm.running_var.data = get_weights(layer.running_var.data, indexes, dim, onedim=True, running_stat=True)
        if global_print: print("Getting new norm running_var layer. Shape: ", new_norm.running_var.data.shape)

    return new_norm


if __name__ == '__main__':
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
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--single-cls', action='store_true', help='treat as single-class dataset')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--merge', action='store_true', help='use Merge NMS')
    parser.add_argument('--verbose', action='store_true', help='report mAP by class')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--cfg', type=str, default='cfg/yolov4_kitti.cfg', help='*.cfg path')
    parser.add_argument('--names', type=str, default='data/kitti.names', help='*.cfg path')
    #parser.add_argument('--save-path', type=str, default='sandbox/DatasetForPruning')
    parser.add_argument('--save-path', type=str, default='sandbox/deleteme')

    parser.add_argument('--test-cases', type=int, default=5000)

    opt = parser.parse_args()
    opt.save_json |= opt.data.endswith('coco.yaml')
    opt.data = check_file(opt.data)  # check file

    # opt.weights = "/home/blanka/YOLOv4_Pruning/weights/yolov4_kitti.weights"

    alphas = np.arange(0.0, 2.3, 0.1).tolist()
    alphas = [float("{:.2f}".format(x)) for x in alphas]

    yolo_layers = [138, 148, 149, 160]
    glob_dims = [0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1,
                 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1,
                 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 1, 0, 1,
                 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0,
                 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1]
    print("len dims ", len(glob_dims))
    dim = 0

    # Load data
    with open(opt.data) as f:
        data = yaml.load(f, Loader=yaml.FullLoader)  # model dict
    val_img_path = data['val_img']
    val_label_path = data['val_label']

    dataloader = create_dataloader(val_img_path, val_label_path, opt.img_size, opt.batch_size, 32, opt,
                                   hyp=None, augment=False, cache=False, pad=0.5, rect=True)[0]
    # Load model
    model = Darknet(opt.cfg).to('cuda')

    # ckpt = torch.load(opt.weights)
    # state_dict = {k: v for k, v in ckpt['model'].items() if model.state_dict()[k].numel() == v.numel()}
    # model.load_state_dict(state_dict, strict=False)

    model.load_darknet_weights(opt.weights)

    # model.save_weights("weights/yolov4_kitti.weights")

    # model_save_path = "/home/blanka/YOLOv4_Pruning/sandbox/" + "kitti_params.txt"
    # with open(model_save_path, 'w') as f:
    #    for name, p in model.named_parameters():
    #        f.write(str(name) + " " + str(p.shape) + "\n" + str(p) + "\n")

    # Get metrics before pruning

    results, _, _ = test(
        opt.data,
        batch_size=opt.batch_size,
        imgsz=opt.img_size,
        conf_thres=opt.conf_thres,
        iou_thres=opt.iou_thres,
        save_json=False,  # save json
        single_cls=opt.single_cls,
        augment=opt.augment,
        verbose=opt.verbose,
        model=model,
        opt=opt,
        called_directly=True)

    prec, rec, map = results[0], results[1], results[2]
    prec_before, rec_before, map_before = prec, rec, map

    print("map", map)

    ##
    network_size = len(model.module_list)
    network_param_nmb = sum([param.nelement() for param in model.parameters()])

    for test_case in range(opt.test_cases):
        break

        start_time_test_case = time.time()

        if test_case < 3091:
            continue

        num_samples = 1
        mu = 0
        sigma = 0.4
        unsorted_probs = generate_gaussian_probs(mu, sigma, len(alphas))

        save_path = opt.save_path + F"/test_{test_case}.txt"
        model = Darknet(opt.cfg).to('cuda')
        model.load_darknet_weights(opt.weights)

        for layer_index in range(network_size):

            start_time = time.time()

            if layer_index in [148, 149]:
                continue

            print("maind", layer_index)

            layer_param_nmb_before = sum(
                [param.nelement() for name, param in model.named_parameters() if "." + str(layer_index) + "." in name])
            param_nmb_before = sum([param.nelement() for param in model.parameters()])

            module_def = model.module_defs[layer_index]
            if module_def["type"] in ["route", "shortcut", "upsample", "maxpool", "yolo"] or glob_dims[
                layer_index] == 1:
                continue
            layer = model.module_list[layer_index][0]  # only if convolutional

            # Generate random alpha instances
            probs = sort_gaussian_probs(unsorted_probs, len(alphas), layer_index, network_size)
            print(probs)
            alpha = sorted(random.choices(alphas, weights=probs, k=num_samples))[0]

            skip = random.choice([1, 2, 3, 4, 5, 6])

            if layer_index in [11, 24, 55]:
                while alpha > 0.2:
                    alpha = sorted(random.choices(alphas, weights=probs, k=num_samples))[0]
            # elif layer_index in [86, 115]:
            #    alpha = 2.2
            elif layer_index in [123, 125]:
                while alpha > 0.8:
                    alpha = sorted(random.choices(alphas, weights=probs, k=num_samples))[0]
            elif layer_index > 134 and layer_index < 155:
                while alpha > 1.3:
                    alpha = sorted(random.choices(alphas, weights=probs, k=num_samples))[0]
            elif layer_index > 154 or layer_index in [86, 115]:
                while alpha < 1.3:
                    alpha = sorted(random.choices(alphas, weights=probs, k=num_samples))[0]
            else:
                if layer_index > 100 and layer_index % skip == 0:
                    while alpha > 0.3:
                        alpha = sorted(random.choices(alphas, weights=probs, k=num_samples))[0]
                else:
                    alpha = 0.0

            """
            if layer.in_channels == layer.out_channels:
                alpha = 0.0
            elif layer.in_channels > layer.out_channels:
                if layer.in_channels == 1024:
                    while alpha > 1.8:
                        alpha = sorted(random.choices(alphas, weights=probs, k=num_samples))[0]
                elif layer.in_channels == 512:
                    while alpha > 0.6:
                        alpha = sorted(random.choices(alphas, weights=probs, k=num_samples))[0]
                elif layer.in_channels == 256:
                    while alpha > 0.1:
                        alpha = sorted(random.choices(alphas, weights=probs, k=num_samples))[0]
                else:
                    alpha = 0.0
            elif layer.out_channels == 1024:
                alpha = 2.2
            elif layer.out_channels == 512:
                while alpha > 1.8:
                    alpha = sorted(random.choices(alphas, weights=probs, k=num_samples))[0]
            elif layer.out_channels in [128, 256] :
                while alpha > 0.2:
                    alpha = sorted(random.choices(alphas, weights=probs, k=num_samples))[0]
            else:
                alpha = 0.0
            """

            # for alpha in test_alphas:

            # model_save_path = "/home/blanka/YOLOv4_Pruning/sandbox/pruning_dataset/" + F"params_before_{layer_index}_{alpha}.txt"
            # with open(model_save_path, 'w') as f:
            #    for name, p in model.named_parameters():
            #        f.write(str(name) + " " + str(p.shape) + "\n" + str(p) + "\n")

            model, parser = prune_network(model, yolo_layers, layer_index, alpha, dataset_make=True)
            model.to('cuda')

            # model_save_path = "/home/blanka/YOLOv4_Pruning/sandbox/pruning_dataset/" + F"model_{layer_index}_{alpha}.txt"
            # with open(model_save_path, 'w') as f:
            #    f.write(str(model))

            # model_save_path = "/home/blanka/YOLOv4_Pruning/sandbox/pruning_dataset/" + F"params_{layer_index}_{alpha}.txt"
            # with open(model_save_path, 'w') as f:
            #    for name, p in model.named_parameters():
            #        f.write(str(name) + " " + str(p.shape) + "\n" + str(p) + "\n")

            # if layer_index != 0 and prec_before == 0.00000000 and rec_before == 0.00000000 and map_before == 0.00000000:
            #    print("Skipped before everything is 0.")
            #    prec_after, rec_after, map_after = prec_before, rec_before, map_before
            #    prec_before, rec_before, map_before = prec_after, rec_after, map_after

            if alpha == 0.0:
                prec_after, rec_after, map_after = prec_before, rec_before, map_before
                prec_before, rec_before, map_before = prec_after, rec_after, map_after

            else:

                results, _, _ = test(
                    opt.data,
                    # opt.weights,
                    batch_size=opt.batch_size,
                    imgsz=opt.img_size,
                    conf_thres=opt.conf_thres,
                    iou_thres=opt.iou_thres,
                    save_json=False,  # save json
                    single_cls=opt.single_cls,
                    augment=opt.augment,
                    verbose=opt.verbose,
                    model=model,
                    opt=opt,
                    called_directly=True)

                prec_after, rec_after, map_after = results[0], results[1], results[2]
                prec_before, rec_before, map_before = prec_after, rec_after, map_after

            param_nmb_after = sum([param.nelement() for param in model.parameters()])
            layer_param_nmb_after = sum(
                [param.nelement() for name, param in model.named_parameters() if "." + str(layer_index) + "." in name])

            metrics = [
                layer_index,
                "%.1f" % alpha,
                parser['dim'],
                parser['in_ch'],
                parser['out_ch'],
                parser['kernel'],
                parser['stride'],
                parser['pad'],
                parser['nmb_of_pruned_ch'],
                network_param_nmb,
                param_nmb_before,
                param_nmb_after,
                layer_param_nmb_before,
                layer_param_nmb_after,
                "%.4f" % round((1 - param_nmb_after / network_param_nmb) * 100, 4),
                "%.4f" % round(prec, 4),
                "%.10f" % round(prec_after, 10),
                "%.4f" % round(rec, 4),
                "%.10f" % round(rec_after, 10),
                "%.4f" % round(map, 4),
                "%.10f" % round(map_after, 10)
            ]

            elapsed_time = time.time() - start_time
            print("Runtime of one layer pruning: ", elapsed_time)
            print(metrics)
            metrics = str(metrics).replace('[', '').replace(']', '').replace(',', '').replace('\'', '')

            with open(save_path, 'a+') as f:
                f.write(str(metrics) + "\n")

            if layer_index == 159:
                prec_before, rec_before, map_before = prec, rec, map

            """
            metrics = [
                layer_index,
                "%.1f" % alpha,
                parser['dim'],
                parser['in_ch'],
                parser['out_ch'],
                parser['kernel'],
                parser['stride'],
                parser['pad'],
                parser['nmb_of_pruned_ch'],
                network_param_nmb,
                param_nmb_before,
                param_nmb_after,
                layer_param_nmb_before,
                layer_param_nmb_after
            ]
            with open("sandbox/gaussian.txt", 'a+') as f:
                f.write(str(layer_index) +str(probs) + str(alpha) + "\n")
            """
        print("Runtime of one testcase: ", time.time() - start_time_test_case)