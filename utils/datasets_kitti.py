"""
This script does not belong to this project (YOLOv4 pruning), it's just an example of implementing ListDataset class.
"""


import glob
import random
import os
import os.path as path
import sys
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
import imageio

from torch.utils.data import Dataset
import torchvision.transforms as transforms

def convert_kitti_to_coco_ids(id):

    if (id == "0.000" or id == "0" ): # car
        id = "2.000"
    elif (id == "1.000" or id == "1" or id == "2.000" or id == "2" ): # truck
        id = "7.000"
    elif (id == "3.000" or id == "3" or id == "4.000" or id == "4"): # person
        id = "0.000"
    elif (id == "5.000" or id == "5"):  # bike
        id = "1.000"
    elif (id == "6.000" or id == "6" ): # tram -> train
        id = "5.000"
    elif (id == "7.000" or id == "7" ): # misc
        id = "-1.000"

    return id


def convert_m86_to_coco_ids(id):

    if (id == "0.000" or id == "1.000" or id == "0" or id == "1"):
        id = "2.000"

    return id

def pad_to_square(img, pad_value):
    c, h, w = img.shape
    dim_diff = np.abs(h - w)
    # (upper / left) padding and (lower / right) padding
    pad1, pad2 = dim_diff // 2, dim_diff - dim_diff // 2
    # Determine padding
    pad = (0, 0, pad1, pad2) if h <= w else (pad1, pad2, 0, 0)
    # Add padding
    img = F.pad(img, pad, "constant", value=pad_value)

    return img, pad


def resize(image, size):
    image = F.interpolate(image.unsqueeze(0), size=size, mode="nearest").squeeze(0)
    return image


def random_resize(images, min_size=288, max_size=448):
    new_size = random.sample(list(range(min_size, max_size + 1, 32)), 1)[0]
    images = F.interpolate(images, size=new_size, mode="nearest")
    return images


class ImageFolder(Dataset):
    def __init__(self, folder_path, img_size=416):
        self.files = sorted(glob.glob("%s/*.*" % folder_path))
        self.img_size = img_size

    def __getitem__(self, index):
        img_path = self.files[index % len(self.files)]
        # Extract image as PyTorch tensor
        img = transforms.ToTensor()(Image.open(img_path))
        # Pad to square resolution
        img, _ = pad_to_square(img, 0)
        # Resize
        img = resize(img, self.img_size)

        return img_path, img

    def __len__(self):
        return len(self.files)


class ListDataset(Dataset):
    def __init__(self, anns_path, imgs_path, img_size=416, multiscale=True, normalized_labels=True, kitti2coco=False, m862coco=False):

        self.imgs_path = imgs_path
        self.anns_path = anns_path
        self.imgs_path2 = imgs_path2
        self.anns_path2 = anns_path2
        if imgs_path2 != "":
            self.img_files = [x for x in os.listdir(imgs_path) if x.endswith(".png")] + [x for x in os.listdir(imgs_path2) if x.endswith(".png")]
        else:
            self.img_files = [x for x in os.listdir(imgs_path) if x.endswith(".png")]

        self.label_files = [
            path.replace("images", "labels").replace(".png", ".txt").replace(".jpg", ".txt")
            for path in self.img_files]

        self.img_size = img_size
        self.orig_size = 1242
        self.max_objects = 10
        self.scale = dist_scale
        self.multiscale = multiscale
        self.normalized_labels = normalized_labels
        self.min_size = self.img_size - 3 * 32
        self.max_size = self.img_size + 3 * 32
        self.batch_count = 0
        self.kitti2coco = kitti2coco
        self.m862coco = m862coco

    def __getitem__(self, index):

        # ---------
        #  Image
        # ---------

        img_file = self.img_files[index % len(self.img_files)].rstrip()
        if path.exists(self.imgs_path + "\\" + img_file):
            img_path = self.imgs_path + "\\" + img_file
        elif path.exists(self.imgs_path2 + "\\" + img_file):
            img_path = self.imgs_path2 + "\\" + img_file
        else:
            print("Error: image file path does not exist.")


        imio = imageio.imread(img_path)

        img = transforms.ToTensor()(imio)

        # Handle images with less than three channels
        if len(img.shape) != 3:
            img = img.unsqueeze(0)
            img = img.expand((3, img.shape[1:]))

        _, h, w = img.shape
        self.orig_size = h
        h_factor, w_factor = (h, w) if self.normalized_labels else (1, 1)


        # Pad to square resolution
        img, pad = pad_to_square(img, 0)
        _, padded_h, padded_w = img.shape


        # ---------
        #  Label
        # ---------

        label_file = self.label_files[index % len(self.img_files)].rstrip()
        label_path = self.anns_path + "\\" + label_file
        if path.exists(self.anns_path + "\\" + label_file):
            label_path = self.anns_path + "\\" + label_file
        elif path.exists(self.anns_path2 + "\\" + label_file):
            label_path = self.anns_path2 + "\\" + label_file
        else:
            print("Error: label file path does not exist.")
        #print(label_path)
        add_targets = []

        with open(label_path, 'r') as f:
            for line in f:

                split = line.split(" ")

                if len(split) != 6:
                    print("Continued")
                    continue

                id, x, y, w, h, dist = line.split(" ")

                if self.kitti2coco:
                    id = convert_kitti_to_coco_ids(id)
                elif self.m862coco:
                    id = convert_m86_to_coco_ids(id)

                if (id != "-1.000"):
                    label = [float(id), float(x), float(y), float(w), float(h), float(dist)]
                    add_targets.append(label)

        add_targets = np.array(add_targets)
        add_targets = add_targets.reshape(-1, 6)

        add_targets = torch.from_numpy(add_targets)

        targets = torch.zeros((len(add_targets), 7))
        targets[:, 1:] = add_targets
        if not self.normalized_labels:
            targets[:, 2:6] /= self.orig_size
        targets[:, 6] /= self.scale

        return img_path, img, targets

    def collate_fn(self, batch):
        paths, imgs, targets = list(zip(*batch))
        # Remove empty placeholder targets
        targets = [boxes for boxes in targets if boxes is not None]
        # Add sample index to targets
        for i, boxes in enumerate(targets):
            boxes[:, 0] = i
        targets = torch.cat(targets, 0)
        """
        # Selects new image size every tenth batch
        if self.multiscale and self.batch_count % 10 == 0:
            self.img_size = random.choice(range(self.min_size, self.max_size + 1, 32))
        """

        # Resize images to input shape
        imgs = torch.stack([resize(img, self.img_size) for img in imgs])

        self.batch_count += 1
        return paths, imgs, targets

    def __len__(self):
        return len(self.img_files)