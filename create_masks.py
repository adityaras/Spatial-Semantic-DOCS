from pycocotools.coco import COCO
import numpy as np
import random
import os
import cv2
import csv
from shutil import copyfile
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm


csv_base = "./dir_csvs/"
image_base = "./data/images/"
gt_base = "./data/gt/"
dataDir='./COCOdataset2017'
dataType='train'
annFile='{}/annotations/stuff_{}2017.json'.format(dataDir,dataType)
coco=COCO(annFile)

def create_mask(x):
    name = f"{gt_base}{x[-1]}/{x[1].split('/')[-1]}"
    img = coco.loadImgs(int(x[0]))[0]
    catIds = coco.getCatIds(catNms=[x[-1]]) 
    annIds = coco.getAnnIds(imgIds=img['id'], catIds=catIds, iscrowd=None)
    anns = coco.loadAnns(annIds)
    mask = np.zeros((img['height'],img['width']))
    for i in range(len(anns)):
        mask = np.maximum(coco.annToMask(anns[i])*255, mask)
    cv2.imwrite(name,mask)
    copyfile(x[1],f"{image_base}{x[-1]}/{x[1].split('/')[-1]}")

animal_csvs = os.listdir(csv_base)

for animal_csv in animal_csvs:
    animal = animal_csv[:-4]
    with open(csv_base+ animal_csv, "r") as f:
        reader = csv.reader(f)
        data = list(reader)
    print(f"\t Processing {animal}, please wait.")
    print(data)
    # exit()
    for x in tqdm(data):
        if x:
            name = f"{gt_base}{x[-1]}/{x[1].split('/')[-1]}"
            img = coco.loadImgs(int(x[0]))[0]
            catIds = coco.getCatIds(catNms=[x[-1]]) 
            annIds = coco.getAnnIds(imgIds=img['id'], catIds=catIds, iscrowd=False)
            anns = coco.loadAnns(annIds)
            mask = np.zeros((img['height'],img['width']))
            for i in range(len(anns)):
                mask = np.maximum(coco.annToMask(anns[i])*255, mask)
            cv2.imwrite(name,mask)
            # print(x[1],f"{image_base}{x[-1]}/{x[1].split('/')[-1]}")
            copyfile(x[1],f"{image_base}{x[-1]}/{x[1].split('/')[-1]}")

    # create_mask(data[0])
    # with ProcessPoolExecutor(max_workers=1) as executor:
    #     tqdm((executor.map(create_mask, data)), total=len(data))
    # break