### For toy dataset
# This file is used to get captions from the json file
# allcaptions(image_id, data) returns all captions of the image of that image_id
# onerandomcaption(image_id, data) returns one random caption of the image of that image_id
# image_id is the id of the image, which is the name of the image file
# data is the json file
# there are two json file: captions_train2017.json and captions_val2017.json
# Plese open the json file and put the data in the variable data
# here is an example:
# 
# import os
# import toydataset.getCaps as getCaps
# import json
# dir_path = './toydataset/images/'
# all_file_name = os.listdir(dir_path)
# with open('toydataset/captions/captions_train2017.json') as f:
#     data = json.load(f)
#     for i in range(len(all_file_name)):
#         id = int(all_file_name[i].split('.')[0])
#         caps = getCaps.allcaptions(id,data)
#         cap = getCaps.onerandomcaption(id,data)
#         print(caps, cap)
#
# This would print out all captions and one random caption of each image in the toy dataset

import random

def allcaptions(image_id, data):
    captions = []
    anno = data['annotations']
    for i in range(len(anno)):
        if anno[i]['image_id'] == image_id:
            captions.append(anno[i]['caption'])
    return captions

def onerandomcaption(image_id, data):
    captions = []
    anno = data['annotations']
    for i in range(len(anno)):
        if anno[i]['image_id'] == image_id:
            captions.append(anno[i]['caption'])
    return random.choice(captions)
