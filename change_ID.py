import os
import numpy as np


def bbox_to_xywh(x1, x2, y1, y2):
    """
    Converts the KITTI bounding box format (left top, right bottom)
    to YOLO bounding box format (center x, center y, width, height)
    """
    W, H = 1242, 375

    x1 = float("{:.2f}".format(float(x1)))
    x2 = float("{:.2f}".format(float(x2)))
    y1 = float("{:.2f}".format(float(y1)))
    y2 = float("{:.2f}".format(float(y2)))

    xc = ((x1 + x2) / 2) / W
    yc = ((y1 + y2) / 2) / H
    w = (x2 - x1) / W
    h = (y2 - y1) / H

    return xc, yc, w, h


def get_numID_for_kitti(strID):

    if strID == 'Car':
        numID = '0'
    elif strID == 'Van':
        numID = '1'
    elif strID == 'Truck':
        numID = '2'
    elif strID == 'Pedestrian':
        numID = '3'
    elif strID == 'Person_sitting':
        numID = '4'
    elif strID == 'Cyclist':
        numID = '5'
    elif strID == 'Tram':
        numID = '6'
    elif strID == 'Misc':
        numID = '7'
    elif strID == 'DontCare':
        numID = '8'

    return numID

def get_new_label_file(folder_in, folder_out):

    txt_files = [x for x in os.listdir(folder_in) if x.endswith(".txt")]

    # Get the number of txt files in the folder
    num_imgs = len(txt_files)


    # Id = -1 is the indicator of non-objects, so this is the default id
    # labels[:, :, 0] = -1

    for i, fname in enumerate(txt_files):

        file = os.path.join(folder_in, fname)
        file_out = os.path.join(folder_out, fname)

        with open(file_out, 'w') as f_out:

            with open(file, 'r') as f:

                for j, line in enumerate(f):

                    obj_id, _, _, _, x1, y1, x2, y2, _, _, _, _, _, _, _ = line.split(' ')

                    numID = get_numID_for_kitti(obj_id)

                    x, y, w, h = bbox_to_xywh(float(x1), float(x2), float(y1), float(y2))

                    f_out.write(str(numID) + ' ' + str(x) + ' ' + str(y) + ' ' + str(w) + ' ' + str(h) + '\n')


if __name__ == '__main__':

    folder_in = "/home/blanka/Datasets/KITTI/labels_original"
    folder_out = "/home/blanka/Datasets/KITTI/labels"

    get_new_label_file(folder_in, folder_out)
