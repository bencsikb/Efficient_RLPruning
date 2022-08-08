import os
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
# import torchvision.transforms as transforms
from PIL import Image
from os import path



def bbox_resize(path, orig_size):
    """
    Resize the bounding boxes on the image to fit the resized image

    Input: Original bounding box parameters
           Original image size
           New image size
    Output: Resized bounding box parameters [(max number of objects in the image) x (x, y, w, h)]
    """
    labels = []
    with open (path, 'r') as f:
        for j, line in enumerate(f):
            obj_id, x, y, w, h = line.split(' ')
            x, y, w, h = float(x) / orig_size[0], float(y) / orig_size[1], float(w) / orig_size[0], float(h) / orig_size[1]
            labels.append([int(obj_id), x, y, w, h])

    labels = torch.Tensor(labels)
    labels.reshape(-1, 5)

    return labels


def save_bboxes(out_file, labels):
    """
    Save th annotation of the sized images to the output folder.

    Input: Path of the folder where the annotations have to be saved
           Names of the output files
           Numpy array of object IDs [numImages x maxNumObj x 1]
           Numpy array of bounding boxes [numImages x maxNumObj x 4]
           Numpy array of distances [numImages x maxNumObj x 1]

    """


    with open(out_file, 'w') as f:
        for i in range(labels.shape[0]):
            #print(labels[i].numpy()[0])
            id, x, y, w, h = labels[i].numpy()[0], labels[i].numpy()[1], labels[i].numpy()[2], labels[i].numpy()[3], labels[i].numpy()[4]
            id, x, y, w, h = round(id,3), round(x,3), round(y,3), round(w,3), round(h,3),
            f.write(str(id) + " " + str(x) + " " + str(y) + " "+ str(w) + " "+ str(h) + "\n")
            #np.savetxt(f_out, labels[i, :, :], delimiter=" ", fmt='%.3f')


# In[10]:


folder_imgs = "/home/blanka/Dataset/val2017/val2017_small"
folder_annots = "/home/blanka/Dataset/val_annots_all"
folder_annots_res = "/home/blanka/Dataset/val2017_annots_normalized/"

img_files = [x for x in os.listdir(folder_imgs) if x.endswith(".jpg")]


for i, fname in enumerate(img_files):

    img_file_path = os.path.join(folder_imgs, fname)
    print(img_file_path)
    img = cv2.imread(img_file_path)
    img_size = img.shape[0:2]
    print(img_size)

    #label_file_path = img_file_path.replace('val2017/val2017_small', 'val_annots_all').replace('.jpg', '.txt')
    label_file_path = os.path.join(folder_annots, fname).replace('.jpg', '.txt')
    labels = bbox_resize(label_file_path, img_size)

    out_file_path = os.path.join(folder_annots_res, fname).replace('.jpg', '.txt')
    save_bboxes(out_file_path, labels)