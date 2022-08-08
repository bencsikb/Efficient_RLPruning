import torch
import numpy as np
import glob
import os
import shutil
import random
import re

def normalize(x, x_min, x_max):
    # Between -1 and 1

    x = float(x)
    x = 2*((x - x_min) / (x_max - x_min))-1
    return x


def sorted_nicely( l ):
    """ Sort the given iterable in the way that humans expect."""
    convert = lambda text: int(text) if text.isdigit() else text
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ]
    return sorted(l, key = alphanum_key)


def convert2(folderIN, folderDATA, folderLABEL, rows, cols):

    logfile = "/home/blanka/YOLOv4_Pruning/sandbox/processed.txt"
    global_cnt = 0

    files = [x for x in os.listdir(folderIN) if x.endswith(".txt")]
    files = sorted_nicely(files)
    print(files)

    datas = []

    for file in files:

        # Save the processed files
        with open(logfile, 'a') as lf:
            lf.write(str(file)+'\n')

        row_cnt = 0

        path = os.path.join(folderIN, file)
        num_lines = sum(1 for line in open(path))

        # Handle not finished file generations
        if not ((num_lines == 44) or (num_lines == 46)):
            #print(f"Wrong number of lines: {num_lines}")
            continue

        data = torch.full((rows, cols), -1.0)
        label = torch.zeros([1,2]) # error, sparsity

        with open(path, "r") as f:

            prev_spars = -1.0
            for line_index, line in enumerate(f):

                index, alpha, _, in_ch, out_ch, kernel, stride, pad, _, _, _, _, pnmb_bef, pnmb_after, pruned_perc, _, _, _, _, mAPb, mAPa = line.split(" ")

                if num_lines == 46:
                    if line_index == 41 or line_index == 42:
                        print("Skipped \n")
                        print(index)
                        continue

                data[0, row_cnt] = normalize(alpha, 0, 2.2)
                data[1, row_cnt] = normalize(in_ch, 0, 1024)
                data[2, row_cnt] = normalize(out_ch, 0, 1024)
                data[3, row_cnt] = normalize(kernel, 0, 3)
                data[4, row_cnt] = normalize(stride, 0, 2)
                data[5, row_cnt] = normalize(pad, 0, 1)
                data[6, row_cnt] = prev_spars

                next_spars = normalize(float(pruned_perc), 0, 100)  # percent of pruned params from the whole network
                error = normalize(1.0 - (float(mAPa) / float(0.6850150267)), 0, 1)

                label[0,0], label[0,1] = error, next_spars

                prev_spars = next_spars
                row_cnt += 1

                flags = []
                if len(datas):
                    print("if")
                    flags = [False if (False in (np.asarray(data) == np.asarray(d))) else True for d in datas]

                    if not (True in flags):
                        print("foriffalse")

                        datas.append(data.tolist())
                        data_save_path = os.path.join(folderDATA, str(global_cnt) + ".txt")
                        error_save_path = os.path.join(folderLABEL, str(global_cnt) + ".txt")
                        np.savetxt(data_save_path, data.numpy())
                        np.savetxt(error_save_path, label.numpy())
                        global_cnt += 1
                else:
                    print("else")
                    datas.append(data.tolist())
                    data_save_path = os.path.join(folderDATA, str(global_cnt) + ".txt")
                    error_save_path = os.path.join(folderLABEL, str(global_cnt) + ".txt")
                    np.savetxt(data_save_path, data.numpy())
                    np.savetxt(error_save_path, label.numpy())
                    global_cnt += 1

    print(F"End of data conversion. Generated data files: {global_cnt}")




def convert(folderIN, folderDATA, folderERROR, rows, cols):

    """
    Converts the firstly genarated data to training data format.

    :param folderIN: /home/blanka/YOLOv4_Pruning/sandbox/DatasetForPruning
    :param folderDATA: /home/blanka/Datasets/PruningDataset/data
    :param folderERROR: /home/blanka/Datasets/PruningDataset/error

    Later, the genarated training data will be split into data/data_val and error/error_val folders.

    """
    global_cnt = 0

    files = [x for x in os.listdir(folderIN) if x.endswith(".txt")]
    files = sorted_nicely(files)
    print(files)

    datas = []
    errors = []

    for file in files:
        path = os.path.join(folderIN, file)
        print(path)
        row_cnt = 0
        num_lines = sum(1 for line in open(path))

        # Handle not finished file generations
        if not ((num_lines == 44) or (num_lines == 46)):
            print(f"Wrong number of lines: {num_lines} \n")
            continue

        data = torch.full((rows, cols), -1.0)
        error_gt = torch.zeros([1])

        with open(path, "r") as f:


            for line_index, line in enumerate(f):

                    index, alpha, _, in_ch, out_ch, kernel, stride, pad, _, _, _, _, pnmb_bef, pnmb_after, pruned_perc, _, _, _, _, mAPb, mAPa = line.split(" ")

                    if num_lines == 46:
                        if line_index == 41 or line_index == 42:
                            print("Skipped \n")
                            print(index)
                            continue

                    #print("alpha before ", alpha)
                    alpha = normalize(alpha, 0, 2.2)
                    #print("alpha after ", alpha)
                    data[0, row_cnt] = alpha
                    data[1, row_cnt] = normalize(in_ch, 0, 1024)
                    data[2, row_cnt] = normalize(out_ch, 0, 1024)
                    data[3, row_cnt] = normalize(kernel, 0, 3)
                    data[4, row_cnt] = normalize(stride, 0, 2)
                    data[5, row_cnt] = normalize(pad, 0, 1)
                    data[6, row_cnt] = normalize(1.0-float(pnmb_after)/float(pnmb_bef), 0, 1)  # percent of pruned params from current layer
                    data[7, row_cnt] = normalize(float(pruned_perc), 0, 100) # percent of pruned params from the whole network

                    error_gt[0] = normalize(1.0 - (float(mAPa) / float(0.6850150267)), 0, 1)

                    row_cnt += 1

                    # Check duplication with torch tensors #
                    datas_tensor = torch.tensor(datas)
                    if datas_tensor is not None:
                        datas_tensor = datas_tensor.view(-1,rows,cols)
                    print(datas_tensor.shape)
                    print(data.unsqueeze(0).shape)
                    flag = True
                    for d in datas_tensor:
                        #print("d size",d.unsqueeze(0).shape)
                        if torch.allclose(d.unsqueeze(0).cuda(), data.cuda()):
                            flag = False
                    if flag:
                        datas.append(data.tolist())

                    #if data.unsqueeze(0) not in datas_tensor:
                    #    print("added")
                    #    datas.append(data.tolist())


                    """
                    flags = []
                    if len(datas):
                        print("if")
                        for i, d in enumerate(datas):
                            # Check if alpha sequence is equal
                            ## BUG!!! Not only the alpha sequence needs to be checked!!! ##
                            #if False in (np.asarray(data[0,:]) == np.asarray(d)[0]): # if not equal, APPEND
                            if False in (np.asarray(data) == np.asarray(d)): # if not equal, APPEND

                                flags.append(False)
                            else:
                                flags.append(True)

                        if not (True in flags):
                                print("foriffalse")

                                #print(len(datas))
                                #print(np.asarray(data[0, :]) == np.asarray(d)[0])
                                #print(data[0, :], d[0])

                                datas.append(data.tolist())
                                data_save_path = os.path.join(folderDATA, str(global_cnt) + ".txt")
                                error_save_path = os.path.join(folderERROR, str(global_cnt) + ".txt")
                                np.savetxt(data_save_path, data.numpy())
                                np.savetxt(error_save_path, error_gt.numpy())
                                global_cnt += 1
                    else:
                        print("else")
                        datas.append(data.tolist())
                        data_save_path = os.path.join(folderDATA, str(global_cnt) + ".txt")
                        error_save_path = os.path.join(folderERROR, str(global_cnt) + ".txt")
                        np.savetxt(data_save_path, data.numpy())
                        np.savetxt(error_save_path, error_gt.numpy())
                        global_cnt += 1
                    """

    print(F"End of data conversion. Generated data files: {global_cnt}")

def split_dataset(folderIN, folderDATA, folderERROR, k):
    """
    :param folderIN: destination of data files to be split
    :param folderDATA: output destination of validation data files
    :param folderERROR: output destination of validation error files
    :param k: number of filees to be moved into validation dataset
    :return:
    """

    files = [x for x in os.listdir(folderIN) if x.endswith(".txt")]
    tomove = random.choices(files, k=k)
    for file in files:
        if file in tomove:
            src = os.path.join(folderIN, file)
            dst =  os.path.join(folderDATA, file)
            shutil.move(src, dst)
            src = os.path.join(folderIN.replace('data', 'label'), file)
            dst = os.path.join(folderERROR, file)
            shutil.move(src, dst)


def copy_to_old_dset(namefile, folderOLD, lastname):


    with open(namefile, 'r') as f:

        for line in f:
            src = line
            src = src.replace("\n","")
            print(src)
            with open(src, 'r') as srcf:
                data = np.loadtxt(srcf)
                dst = os.path.join(folderOLD, str(lastname))
                with open(dst, 'w') as dstf:
                    np.savetxt(dstf,data)

            # Do this to error files too
            srcE = src.replace("data", "error")
            with open(srcE, 'r') as srcf:
                error = np.loadtxt(srcf)
                dstE = os.path.join(folderOLD.replace("data", "val"), str(lastname))
                with open(dstE, 'w') as dstf:
                    np.savetxt(dstf,error)

            lastname +=1


def reduce(folderIN, folderOUT):
    """
    Delete given amount of training data where the ground truth error is higher than 0.9
    :param folderIN: label folder of dataset to be reduced
    :param folderOUT: label folder of output
    :return:
    """

    high_cnt = 0

    files = [x for x in os.listdir(folderIN) if x.endswith(".txt")]

    for file in files:

        copyflag = False

        src = os.path.join(folderIN, file)
        with open(src, 'r') as f:
            labels = np.loadtxt(f)
        error = labels[0]

        if error <= 0.9: copyflag = True
        elif high_cnt < 11355:
            copyflag = True
            high_cnt += 1

        if copyflag:
            src_label = os.path.join(folderIN, file)
            dst_label = os.path.join(folderOUT, file)
            shutil.copyfile(src_label, dst_label)
            src_data = src_label.replace("label", "data")
            dst_data = dst_label.replace("label", "data")
            shutil.copyfile(src_data, dst_data)


def change_middle(folderIN, folderOUT):
    """
    Change the "middle" of the data files: from -1.0 to actual layer sizes
    :param folderIN: data folder of old files
    :param folderOUT: data folder of new files
    :return:
    """

    middle = [[-9.941406250000000000e-01,-8.750000000000000000e-01,-8.750000000000000000e-01,-8.750000000000000000e-01,-8.750000000000000000e-01,-8.750000000000000000e-01,-7.500000000000000000e-01,-7.500000000000000000e-01,-7.500000000000000000e-01,-7.500000000000000000e-01,-7.500000000000000000e-01,-7.500000000000000000e-01,-7.500000000000000000e-01,-7.500000000000000000e-01,-7.500000000000000000e-01,-5.000000000000000000e-01,-5.000000000000000000e-01,-5.000000000000000000e-01,-5.000000000000000000e-01,-5.000000000000000000e-01,-5.000000000000000000e-01,-5.000000000000000000e-01,-5.000000000000000000e-01,-5.000000000000000000e-01,0.000000000000000000e+00,0.000000000000000000e+00,0.000000000000000000e+00,0.000000000000000000e+00,0.000000000000000000e+00,1.000000000000000000e+00,1.000000000000000000e+00,0.000000000000000000e+00,0.000000000000000000e+00,-5.000000000000000000e-01,-5.000000000000000000e-01,-5.000000000000000000e-01,-7.500000000000000000e-01,-7.500000000000000000e-01,-7.500000000000000000e-01,-5.000000000000000000e-01,-5.000000000000000000e-01,0.000000000000000000e+00,0.000000000000000000e+00,0.000000000000000000e+00],
                [-9.375000000000000000e-01,-8.750000000000000000e-01,-9.375000000000000000e-01,-7.500000000000000000e-01,-8.750000000000000000e-01,-8.750000000000000000e-01,-5.000000000000000000e-01,-7.500000000000000000e-01,-7.500000000000000000e-01,-7.500000000000000000e-01,-7.500000000000000000e-01,-7.500000000000000000e-01,-7.500000000000000000e-01,-7.500000000000000000e-01,-7.500000000000000000e-01,0.000000000000000000e+00,-5.000000000000000000e-01,-5.000000000000000000e-01,-5.000000000000000000e-01,-5.000000000000000000e-01,-5.000000000000000000e-01,-5.000000000000000000e-01,-5.000000000000000000e-01,-5.000000000000000000e-01,1.000000000000000000e+00,0.000000000000000000e+00,0.000000000000000000e+00,0.000000000000000000e+00,0.000000000000000000e+00,0.000000000000000000e+00,0.000000000000000000e+00,1.000000000000000000e+00,-5.000000000000000000e-01,0.000000000000000000e+00,0.000000000000000000e+00,-7.500000000000000000e-01,-5.000000000000000000e-01,-5.000000000000000000e-01,-5.000000000000000000e-01,0.000000000000000000e+00,0.000000000000000000e+00,1.000000000000000000e+00,1.000000000000000000e+00,1.000000000000000000e+00],
                [1.000000000000000000e+00,-3.333333432674407959e-01,-3.333333432674407959e-01,1.000000000000000000e+00,-3.333333432674407959e-01,1.000000000000000000e+00,1.000000000000000000e+00,-3.333333432674407959e-01,1.000000000000000000e+00,1.000000000000000000e+00,1.000000000000000000e+00,1.000000000000000000e+00,1.000000000000000000e+00,1.000000000000000000e+00,1.000000000000000000e+00,1.000000000000000000e+00,-3.333333432674407959e-01,1.000000000000000000e+00,1.000000000000000000e+00,1.000000000000000000e+00,1.000000000000000000e+00,1.000000000000000000e+00,1.000000000000000000e+00,1.000000000000000000e+00,1.000000000000000000e+00,-3.333333432674407959e-01,1.000000000000000000e+00,1.000000000000000000e+00,1.000000000000000000e+00,-3.333333432674407959e-01,-3.333333432674407959e-01,1.000000000000000000e+00,-3.333333432674407959e-01,1.000000000000000000e+00,1.000000000000000000e+00,-3.333333432674407959e-01,1.000000000000000000e+00,1.000000000000000000e+00,1.000000000000000000e+00,1.000000000000000000e+00,1.000000000000000000e+00,1.000000000000000000e+00,1.000000000000000000e+00,1.000000000000000000e+00],
                [0.000000000000000000e+00,0.000000000000000000e+00,0.000000000000000000e+00,1.000000000000000000e+00,0.000000000000000000e+00,0.000000000000000000e+00,1.000000000000000000e+00,0.000000000000000000e+00,0.000000000000000000e+00,0.000000000000000000e+00,0.000000000000000000e+00,0.000000000000000000e+00,0.000000000000000000e+00,0.000000000000000000e+00,0.000000000000000000e+00,1.000000000000000000e+00,0.000000000000000000e+00,0.000000000000000000e+00,0.000000000000000000e+00,0.000000000000000000e+00,0.000000000000000000e+00,0.000000000000000000e+00,0.000000000000000000e+00,0.000000000000000000e+00,1.000000000000000000e+00,0.000000000000000000e+00,0.000000000000000000e+00,0.000000000000000000e+00,0.000000000000000000e+00,0.000000000000000000e+00,0.000000000000000000e+00,0.000000000000000000e+00,0.000000000000000000e+00,0.000000000000000000e+00,0.000000000000000000e+00,0.000000000000000000e+00,0.000000000000000000e+00,0.000000000000000000e+00,0.000000000000000000e+00,0.000000000000000000e+00,0.000000000000000000e+00,0.000000000000000000e+00,0.000000000000000000e+00,0.000000000000000000e+00],
                [1.000000000000000000e+00,-1.000000000000000000e+00,-1.000000000000000000e+00,1.000000000000000000e+00,-1.000000000000000000e+00,1.000000000000000000e+00,1.000000000000000000e+00,-1.000000000000000000e+00,1.000000000000000000e+00,1.000000000000000000e+00,1.000000000000000000e+00,1.000000000000000000e+00,1.000000000000000000e+00,1.000000000000000000e+00,1.000000000000000000e+00,1.000000000000000000e+00,-1.000000000000000000e+00,1.000000000000000000e+00,1.000000000000000000e+00,1.000000000000000000e+00,1.000000000000000000e+00,1.000000000000000000e+00,1.000000000000000000e+00,1.000000000000000000e+00,1.000000000000000000e+00,-1.000000000000000000e+00,1.000000000000000000e+00,1.000000000000000000e+00,1.000000000000000000e+00,-1.000000000000000000e+00,-1.000000000000000000e+00,1.000000000000000000e+00,-1.000000000000000000e+00,1.000000000000000000e+00,1.000000000000000000e+00,-1.000000000000000000e+00,1.000000000000000000e+00,1.000000000000000000e+00,1.000000000000000000e+00,1.000000000000000000e+00,1.000000000000000000e+00,1.000000000000000000e+00,1.000000000000000000e+00,1.000000000000000000e+00]]


    files = [x for x in os.listdir(folderIN) if x.endswith(".txt")]

    for file in files:

        src = os.path.join(folderIN, file)
        with open(src, 'r') as f:
            data = np.loadtxt(f)
        data[1:6] = middle

        dst = os.path.join(folderOUT, file)
        with open(dst, 'w') as f:
            np.savetxt(dst, data)


if __name__ == '__main__':

    folderIN = '/home/blanka/YOLOv4_Pruning/sandbox/DatasetForPruning'
    #folderIN =  '/home/blanka/Datasets/PruningDataset/data3_val'
    folderOUT =  '/home/blanka/Datasets/PruningDataset/data3_mod_val'
    folderDATA = '/home/blanka/Datasets/PruningDataset/data3_val'
    folderLABEL = '/home/blanka/Datasets/PruningDataset/label3_val'

    namefile =  "/home/blanka/YOLOv4_Pruning/sandbox/tomove.txt"
    folderOLD = '/home/blanka/Datasets/PruningDataset/trial'
    lastname = 37803


    #convert2(folderIN, folderDATA, folderLABEL, 7, 44)
    convert(folderIN, "","", 8, 44)

    #split_dataset(folderIN, folderDATA, folderLABEL, k=1)
    #copy_to_old_dset(namefile, folderOLD, lastname)
    #reduce(folderIN, folderOUT)
    #change_middle(folderIN, folderOUT)