
from pycocotools.coco import COCO
import numpy as np
# import skimage.io as io
import random
import os
import cv2
import csv
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

dataDir='./COCOdataset2017'
dataType='train'
annFile='{}/annotations/stuff_{}2017.json'.format(dataDir,dataType)

# Initialize the COCO api for instance annotations
coco=COCO(annFile)
d = [{"supercategory": "textile", "id": 92, "name": "banner"}, {"supercategory": "textile", "id": 93, "name": "blanket"}, {"supercategory": "plant", "id": 94, "name": "branch"}, {"supercategory": "building", "id": 95, "name": "bridge"}, {"supercategory": "building", "id": 96, "name": "building-other"}, {"supercategory": "plant", "id": 97, "name": "bush"}, {"supercategory": "furniture-stuff", "id": 98, "name": "cabinet"}, {"supercategory": "structural", "id": 99, "name": "cage"}, {"supercategory": "raw-material", "id": 100, "name": "cardboard"}, {"supercategory": "floor", "id": 101, "name": "carpet"}, {"supercategory": "ceiling", "id": 102, "name": "ceiling-other"}, {"supercategory": "ceiling", "id": 103, "name": "ceiling-tile"}, {"supercategory": "textile", "id": 104, "name": "cloth"}, {"supercategory": "textile", "id": 105, "name": "clothes"}, {"supercategory": "sky", "id": 106, "name": "clouds"}, {"supercategory": "furniture-stuff", "id": 107, "name": "counter"}, {"supercategory": "furniture-stuff", "id": 108, "name": "cupboard"}, {"supercategory": "textile", "id": 109, "name": "curtain"}, {"supercategory": "furniture-stuff", "id": 110, "name": "desk-stuff"}, {"supercategory": "ground", "id": 111, "name": "dirt"}, {"supercategory": "furniture-stuff", "id": 112, "name": "door-stuff"}, {"supercategory": "structural", "id": 113, "name": "fence"}, {"supercategory": "floor", "id": 114, "name": "floor-marble"}, {"supercategory": "floor", "id": 115, "name": "floor-other"}, {"supercategory": "floor", "id": 116, "name": "floor-stone"}, {"supercategory": "floor", "id": 117, "name": "floor-tile"}, {"supercategory": "floor", "id": 118, "name": "floor-wood"}, {"supercategory": "plant", "id": 119, "name": "flower"}, {"supercategory": "water", "id": 120, "name": "fog"}, {"supercategory": "food-stuff", "id": 121, "name": "food-other"}, {"supercategory": "food-stuff", "id": 122, "name": "fruit"}, {"supercategory": "furniture-stuff", "id": 123, "name": "furniture-other"}, {"supercategory": "plant", "id": 124, "name": "grass"}, {"supercategory": "ground", "id": 125, "name": "gravel"}, {"supercategory": "ground", "id": 126, "name": "ground-other"}, {"supercategory": "solid", "id": 127, "name": "hill"}, {"supercategory": "building", "id": 128, "name": "house"}, {"supercategory": "plant", "id": 129, "name": "leaves"}, {"supercategory": "furniture-stuff", "id": 130, "name": "light"}, {"supercategory": "textile", "id": 131, "name": "mat"}, {"supercategory": "raw-material", "id": 132, "name": "metal"}, {"supercategory": "furniture-stuff", "id": 133, "name": "mirror-stuff"}, {"supercategory": "plant", "id": 134, "name": "moss"}, {"supercategory": "solid", "id": 135, "name": "mountain"}, {"supercategory": "ground", "id": 136, "name": "mud"}, {"supercategory": "textile", "id": 137, "name": "napkin"}, {"supercategory": "structural", "id": 138, "name": "net"}, {"supercategory": "raw-material", "id": 139, "name": "paper"}, {"supercategory": "ground", "id": 140, "name": "pavement"}, {"supercategory": "textile", "id": 141, "name": "pillow"}, {"supercategory": "plant", "id": 142, "name": "plant-other"}, {"supercategory": "raw-material", "id": 143, "name": "plastic"}, {"supercategory": "ground", "id": 144, "name": "platform"}, {"supercategory": "ground", "id": 145, "name": "playingfield"}, {"supercategory": "structural", "id": 146, "name": "railing"}, {"supercategory": "ground", "id": 147, "name": "railroad"}, {"supercategory": "water", "id": 148, "name": "river"}, {"supercategory": "ground", "id": 149, "name": "road"}, {"supercategory": "solid", "id": 150, "name": "rock"}, {"supercategory": "building", "id": 151, "name": "roof"}, {"supercategory": "textile", "id": 152, "name": "rug"}, {"supercategory": "food-stuff", "id": 153, "name": "salad"}, {"supercategory": "ground", "id": 154, "name": "sand"}, {"supercategory": "water", "id": 155, "name": "sea"}, {"supercategory": "furniture-stuff", "id": 156, "name": "shelf"}, {"supercategory": "sky", "id": 157, "name": "sky-other"}, {"supercategory": "building", "id": 158, "name": "skyscraper"}, {"supercategory": "ground", "id": 159, "name": "snow"}, {"supercategory": "solid", "id": 160, "name": "solid-other"}, {"supercategory": "furniture-stuff", "id": 161, "name": "stairs"}, {"supercategory": "solid", "id": 162, "name": "stone"}, {"supercategory": "plant", "id": 163, "name": "straw"}, {"supercategory": "structural", "id": 164, "name": "structural-other"}, {"supercategory": "furniture-stuff", "id": 165, "name": "table"}, {"supercategory": "building", "id": 166, "name": "tent"}, {"supercategory": "textile", "id": 167, "name": "textile-other"}, {"supercategory": "textile", "id": 168, "name": "towel"}, {"supercategory": "plant", "id": 169, "name": "tree"}, {"supercategory": "food-stuff", "id": 170, "name": "vegetable"}, {"supercategory": "wall", "id": 171, "name": "wall-brick"}, {"supercategory": "wall", "id": 172, "name": "wall-concrete"}, {"supercategory": "wall", "id": 173, "name": "wall-other"}, {"supercategory": "wall", "id": 174, "name": "wall-panel"}, {"supercategory": "wall", "id": 175, "name": "wall-stone"}, {"supercategory": "wall", "id": 176, "name": "wall-tile"}, {"supercategory": "wall", "id": 177, "name": "wall-wood"}, {"supercategory": "water", "id": 178, "name": "water-other"}, {"supercategory": "water", "id": 179, "name": "waterdrops"}, {"supercategory": "window", "id": 180, "name": "window-blind"}, {"supercategory": "window", "id": 181, "name": "window-other"}, {"supercategory": "solid", "id": 182, "name": "wood"}, {"supercategory": "other", "id": 183, "name": "other"}]

fc = ["bridge", "clothes", "flower", "house", "skyscraper", "table", "tree", "other"]
filterClasses = [i['name'] for i in d if i["name"] in fc]
print(filterClasses)
# exit()

# Load the categories in a variable
# filterClasses = ['laptop', 'tv', 'cell phone']


# Fetch class IDs only corresponding to the filterClasses
for animal in filterClasses:
    catIds = coco.getCatIds(catNms=[animal]) 
    # Get all images containing the above Category IDs
    imgIds = coco.getImgIds(catIds=catIds)
    # print(imgIds)
    # x = np.random.randint(0,len(imgIds))
    l = []
    print(f"Number of images containing all the {animal} classes: {len(imgIds)}")
    for i in range(len(imgIds)):
        img = coco.loadImgs(imgIds[i])[0]
        # print(imgIds[i],f"{dataDir}/images/{dataType}/{img['file_name']}",img)
        l.append([imgIds[i],f"{dataDir}/images/{dataType}/{img['file_name']}",animal])
        # exit()
        # print(imgIds[i],f"{dataDir}/images/{dataType}/")
        # break
    # break
    with open(f"dir_csvs/{animal}.csv","w") as f:
        write = csv.writer(f)
        write.writerows(l)

    # # print(img)
    # I = io.imread('{}/images/{}/{}'.format(dataDir,dataType,img['file_name']))/255.0

    
    # # plt.imshow(I)
    # # plt.axis('off')
    # annIds = coco.getAnnIds(imgIds=img['id'], catIds=catIds, iscrowd=None)
    # anns = coco.loadAnns(annIds)
    # # print(anns)
    # coco.showAnns(anns)
    # mask = np.zeros((img['height'],img['width']))
    # for i in range(len(anns)):
    #     # print(list(coco.annToMask(anns[i])))
    #     mask = np.maximum(coco.annToMask(anns[i])*255, mask)
    
    # print(list(mask))
    # print(np.unique(mask,return_counts=True))
    # cv2.imshow(",asto",mask/255)
    # cv2.waitKey(0)
    # # plt.show()
    # # plt.show()

    # break
# print(cats)
