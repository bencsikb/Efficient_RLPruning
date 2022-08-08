import torch
import torch.nn as nn
import math
from numpy import linalg as LA

from models.models import *
from utils.layers import *


def choose_channels_to_prune(layer, alpha, dim):

    indexes = []
    norms = []

    if dim == 0:
        # Count norm for every channel
        for i in range(layer.out_channels):
            #print(layer.weight.data.shape)
            #print(i, " ", layer.weight.data[i, :, :, :])
            norms.append(LA.norm(layer.weight.data[i, :, :, :]))
            #print("norms ", norms[i])

    else: # dim == 1
        # Count norm for every channel
        for i in range(layer.out_channels):
            #print(layer.weight.data.shape)
            #print(i, " ", layer.weight.data[i, :, :, :])
            norms.append(LA.norm(layer.weight.data[i, :, :, :]))
            #print("norms ", norms[i])


    # Count the average and standard deviation of the channel norms in the layer
    norms = np.asarray(norms)
    norms_avg = np.mean(norms)
    norms_std = np.std(norms)
    print("avg ", norms_avg)
    print("std ", norms_std)

    # Find the indexes of the channels that have to be pruned
    for i, norm in enumerate(norms):
        if (np.absolute(norm) < norms_avg + alpha * norms_std) and (np.absolute(norm) > norms_avg - alpha * norms_std):
            #print("smaller norm ", i, " ", norm)
            indexes.append(i)

    print("len(indexes) ", len(indexes))
    return indexes

def get_route_indexes(dims, routes, index_container, old_output_filters, f):

    indexes = []
    adding = 0

    f.write("old outputs " + str(old_output_filters) + "\n")
    #with open(file, 'a+') as f:
    if True:
        f.write("********** Get route indexes *********\n")
        for k in routes:
            f.write("k " + str(k) + "\n")
            print("k ", k)
            #print(len(index_container[k]))
            #k = k-1
            f.write("dims " + str(dims) + "\n")
            if index_container[k] is not None:
                f.write("len(index_container[k]) " + str(len(index_container[k])) + "\n")
            else:
                f.write("index cont in None \n")

            print(dims)
            print(dims[-1])
            if dims[k-1] == 0:
                f.write("HERE\n")
                print("HERE")

                indexes += list(np.asarray(index_container[k]) + adding)
                f.write("adding "  + str(adding) + "\n")
                #adding = adding + len(index_container[k])
                adding = adding + old_output_filters[k]

        f.write(str(indexes) + "\n")
        if len(indexes):
            f.write("len(indexes) " + str(len(indexes)) + "\n")
        else:
            f.write(" len(indexes) 0 \n")
        f.write("********** ***************\n")
    return indexes

def get_shortcut_indexes(dims, fromm, index_container, output_filters):


    min_channel = output_filters[-1]
    indexes = index_container[-1]

    if output_filters[fromm] < min_channel:
        min_channel = output_filters[fromm]
        indexes = index_container[fromm]

    input_channels = min_channel

    return indexes




def prune_network(network, yolo_layers):
    network = network.cpu()

    # if dim == 0 --> prune the output channels
    # if dim == 1 --> prune the input channels
    dim = 0  # 0: prune corresponding dim of filter weight [out_ch, in_ch, k1, k2]
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

    save_path = "/home/blanka/YOLOv4_Pruning/sandbox/" + "pruning_develop_data.txt"
    with open(save_path, 'w') as f:
    #if True:

        for i in range(network_size):

            sequential_size = len(network.module_list[i])
            module_def = network.module_defs[i]
            f.write("\nmodule_def " + str(module_def["type"]) + str(module_def)+ "\n")

            if module_def["type"] == "route":
                routes = [int(x) for x in module_def["layers"]]
                filters = sum([output_filters[l + 1 if l > 0 else l] for l in routes])
                old_filters = sum([old_output_filters[l + 1 if l > 0 else l] for l in routes])
                indexes = get_route_indexes(dims, routes, index_container, old_output_filters, f)
                alpha = alphas[routes[-1]]
                f.write("route dims" +str(dims)+ "\n")
                #f.write("route alphas"+str(alphas)+ "\n")
                f.write("route output_filters"+str(output_filters)+ "\n")
                f.write("route index lens " + str(index_lens) + "\n")
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
                f.write("shortcut dims"+str(dims)+ "\n")
                #f.write("shortcut alphas"+str(alphas)+ "\n")
                f.write("shortcut output_filters"+str(output_filters)+ "\n")
                f.write("shortcut index lens " + str(index_lens) + "\n")
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
                f.write("yolo dims"+str(dims)+ "\n")
                #f.write("yolo alphas"+str(alphas)+ "\n")
                f.write("yolo output_filters"+str(output_filters)+ "\n")
                dims.append(dims[-1])
                output_filters.append(0)
                old_output_filters.append(0)
                index_container.append([])
                index_lens.append(len(indexes))
                
            elif module_def["type"] == "maxpool" or module_def["type"] == "upsample":
                alpha = alphas[-1]
                f.write("maxpool dims"+str(dims)+ "\n")
                #f.write("maxpool alphas"+str(alphas)+ "\n")
                f.write("maxpool output_filters"+str(output_filters)+ "\n")
                dims[-1] = 0
                dims.append(0)
                output_filters.append(output_filters[-1])
                old_output_filters.append(old_output_filters[-1])
                index_container.append(index_container[-1])
                index_lens.append(len(indexes))
            elif module_def["type"] == "convolutional":

                for j in range(sequential_size):

                    layer = network.module_list[i][j]
                    f.write("Pruning layer " + str(i) + " " + str(layer) + "\n")

                    if isinstance(layer, nn.Conv2d):

                        old_output_filters.append(layer.weight.shape[0])
                        if (i in yolo_layers) and (dim==0):
                        #if i in yolo_layers:
                            output_filters.append(255)
                            dims.append(0) # dont care
                            break
                        elif (i in yolo_layers) and (dim==1):
                            yolo_flag = True


                        if len(dims):
                            dim = dims[-1]
                        else:
                            dim = 0

                        if (dim == 0): # get new alpha
                            alpha = 1
                            indexes = choose_channels_to_prune(layer, alpha, dim)
                            index_container.append(indexes)
                            index_lens.append(len(indexes))
                        else: # dim == 1
                            f.write("else branch \n")
                            f.write(str(prev_indexes))
                            indexes = index_container[-1]
                            index_container.append([])
                            index_lens.append(0)

                        f.write("conv dims "+str( dims)+ "\n")
                        #f.write("conv alphas"+str(alphas)+ "\n")
                        f.write("conv output_filters "+str( output_filters)+ "\n")
                        f.write("conv dim "+str(dim)+ "\n")
                        if indexes is not None:
                            f.write("conv indexes " + str(len(indexes)) + "\n")
                        else:
                            f.write("conv indexes " + str(0) + "\n")
                        f.write("conv index lens " + str(index_lens) + "\n")


                        #if len(indexes):
                        if indexes is not None:

                            new_conv, new_def= get_new_conv(layer, indexes, dim, module_def, yolo_flag)
                            network.module_list[i][j] = new_conv
                            network.module_defs[i] = new_def

                            f.write("New conv layer: " + str(new_conv) + "\n")
                            f.write("New conv def: " + str(new_def) + "\n")


                        dim ^= 1
                        dims.append(dim)
                        print(dims)
                        yolo_flag = False
                        output_filters.append(network.module_defs[i]["filters"])


                    elif isinstance(layer, nn.BatchNorm2d) and (dim == 1):

                        f.write("bnorm dims"+str( dims)+ "\n")
                        f.write("bnorm alphas"+str( alphas)+ "\n")

                        f.write("bnorm dim"+str( dim)+ "\n")
                        f.write("bnorm indexes"+str(indexes)+ "\n")

                        if len(indexes):
                            new_norm = get_new_norm(layer,indexes, dim)
                            network.module_list[i][j] = new_norm

                            f.write("New cbn layer: " + str(new_norm) + "\n")
            #dims.append(dim)
            alphas.append(alpha)

    return network


def get_new_conv(layer, indexes, dim, module_def, yolo_flag):

    print("yolo flag ", yolo_flag)
    if yolo_flag:
        new_bias = False
    else:
        new_bias = layer.bias
        print("new bias ", new_bias)


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
        print("New layer ", new_conv.weight.data.shape)
        #if layer.bias is not None:

        if new_bias:
            new_conv.bias.data = get_weights(layer.bias.data, indexes, dim, onedim=True)
        print("yolo flag ", yolo_flag)

    else:
        in_planes = int(layer.in_channels) - len(indexes)

        new_conv = nn.Conv2d(in_channels=in_planes,
                             out_channels=layer.out_channels,
                             kernel_size=layer.kernel_size,
                             stride=layer.stride,
                             padding=layer.padding,
                             bias=new_bias)

        new_conv.weight.data = get_weights(layer.weight.data, indexes, dim)
        print("New layer ", new_conv.weight.data.shape)
        #if layer.bias is not None:
        if new_bias:
            new_conv.bias.data = get_weights(layer.bias.data, indexes, dim, onedim=True)

        if yolo_flag:
            print("here though")
            new_conv.bias = layer.bias
            new_conv.bias.data = layer.bias.data


    return new_conv, module_def


def get_weights(weights, indexes, dim, onedim=False, running_stat=False):

    if not onedim:
        print("weights.size(dim) ", weights.size(dim))
        print(weights.shape)
        size = list(weights.size())
        new_size = size
        new_size[dim] = weights.size(dim) - len(indexes)
        print(new_size)

        new_weights = torch.zeros(new_size)
        print(new_weights.shape)

        cnt = 0
        for i in range(weights.shape[dim]):
            if i not in indexes:

                if dim == 0:
                    new_weights[cnt, ...] = weights[i, ...]
                else:
                    if weights.shape[2] == 1:
                        new_weights[:, cnt] = weights[:,i]
                    else:
                    #new_weights[:, cnt, :, :] = weights[:, i, :,:]
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
    print("Getting new norm weights layer. Shape: ", new_norm.weight.data.shape)
    if layer.bias is not None:
        new_norm.bias.data = get_weights(layer.bias.data, indexes, 0, onedim=True)
        print("Getting new norm bias layer. Shape: ", new_norm.bias.data.shape)

    if layer.track_running_stats is not None:
        new_norm.running_mean.data = get_weights(layer.running_mean.data, indexes, dim, onedim=True, running_stat=True)
        print("Getting new norm running mean layer. Shape: ", new_norm.running_mean.data.shape)
        new_norm.running_var.data = get_weights(layer.running_var.data, indexes, dim, onedim=True, running_stat=True)
        print("Getting new norm running_var layer. Shape: ", new_norm.running_var.data.shape)

    return new_norm

