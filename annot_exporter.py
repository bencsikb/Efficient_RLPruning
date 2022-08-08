from pycocotools.coco import COCO
import  random
import numpy as np
import os
import skimage.io as io
import cv2
import matplotlib.pyplot as plt

def load_IDs(imgIds_path, delimiter, catIds=False):
    Ids = []

    if catIds:
        with open(imgIds_path, 'r') as input:
            content = input.read()
            content_list = content.split()
            for c in content_list:
                Ids.append(c)


    # open file and read the content into a list
    with open(imgIds_path, 'r') as input:
        content = input.read()
        content_list = content.split(delimiter)

        Ids = content_list

    return Ids

def get_imageIds(annFile, supCats=None):
    coco = COCO(annFile)
    imgIds = []
    true_insts = np.zeros(91)


    if supCats is not None:
        catsForm = coco.loadCats(
            #coco.getCatIds(supNms=['furniture', 'person', 'food', 'appliance', 'indoor', 'electronic', 'kitchen']))
            coco.getCatIds(supNms=supCats)
        )
        cats = [cat['name'] for cat in catsForm]
        #print(cats)

    # If category names are given, get the category IDs
    if (cats != None):
        catIds = []
        for cat in cats:
            catId = coco.getCatIds(catNms=[cat])
            catIds = catIds + catId

    print(len(catIds), catIds)
    catinsts = np.zeros(90)

    # Get all images from each category
    for catId in catIds:
        imgIds = imgIds + coco.getImgIds(catIds=catId)


    # Delete duplicated elements
    res = []
    [res.append(x) for x in imgIds if x not in res]
    imgIds = res

    return imgIds, catIds



def make_annot_files(imgIds, catIds, imgPath=None, annPath=None, annSavePath=None):
    coco = COCO(annPath)
    true_insts = np.zeros(91)
    nocat = []

    for i, imid in enumerate(imgIds):
        if True:

            img_file = coco.loadImgs(imid)[0]
            img_file_name =  img_file['file_name']
            ann_file_name = img_file_name.replace('.jpg', '.txt')
            ann_save_name = annSavePath + ann_file_name
            img_file_path = imgPath + img_file_name
            print(img_file_path)
            if not os.path.exists(ann_save_name):


                img = cv2.imread(img_file_path)
                orig_size = img.shape[0:2]
                plt.imshow(img)
                plt.show()

                print(orig_size)
                print(ann_save_name)

                with open(ann_save_name, "w") as f:
                #if True:
                    annIds = coco.getAnnIds(imgIds=imid, iscrowd=None)
                    objects = 0
                    for annid in annIds:
                        ann = coco.loadAnns(annid)[0]

                        new_cat_id = catIds.index(int(ann['category_id']))
                        x1, y1 = ann['bbox'][0], ann['bbox'][1]
                        w, h = ann['bbox'][2], ann['bbox'][3]

                        x = x1 + w/2
                        y = y1 + h/2

                        x, y, w, h = float(x) / orig_size[1], float(y) / orig_size[0], float(w) / orig_size[1], float(h) / \
                                     orig_size[0]

                        x, y, w, h = round(x, 3), round(y, 3), round(w, 3), round(h, 3)


                        f.write(str(new_cat_id) + " " + str(x) + " " + str(y) + " " + str(w) + " " + str(h) + "\n")

                        objects += 1
            else:
                print("This image has already been normalized")



if __name__ == '__main__':

    annFile = "/home/blanka/Dataset/_annotations/instances_train2017.json"
    imgFolder = "/home/blanka/Dataset/train2017/"

    mySupCats = ['furniture', 'food', 'appliance', 'indoor', 'electronic', 'kitchen', 'person', 'animal', 'vehicle', 'outdoor', 'sports', 'accessory']

    #imgIds, catIds = get_imageIds(annFile, supCats=mySupCats)
    path = "/home/blanka/YOLOv4_Pruning/all_train_img_ids.txt"
    imgIds = load_IDs(path, ", ", catIds=False)

    catIds =  [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 27, 28, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 67, 70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 89, 90]

    #with open(path, 'w') as f:
    #    f.write(str(imgIds))


    print(len(imgIds))

    imgids = []
    for id in imgIds:
        id = int(id)
        imgids.append(id)

    annSavePath = "/home/blanka/Dataset/train2017_annots_normalized/"
    make_annot_files(imgids, catIds, imgPath=imgFolder, annPath=annFile, annSavePath=annSavePath)