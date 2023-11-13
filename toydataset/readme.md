# Toy dataset for coco-stuff
This is a tiny version of the coco-stuff dataset. We use 1000 images from the train2017 folder as our data.
The images is in folder ./images
The segmentation map, called annotation in this case, is in folder ./annotations
The annotation file corresponds to the image file with the same name.
The name of the files is the image's id, which will be used to call its caption.
the captions is in a json file in folder ./captions called captions_train2017.json
# How to use toy dataset
Load the images and their corresponding annotations and the file names as image ids.
Use the image ids to access the captions of the images. 
The function allcaptions(image_id, data) will return all (five) captions of the image.
onerandomcaption(image_id, data) will return one of the captions of the iamge.
The function inputs:
int image_id: this is the image file name.
dic data: this is the data that's in the json file.
# Sample code

import os
import toydataset.getCaps as getCaps
import json
dir_path = './toydataset/images/'
all_file_name = os.listdir(dir_path)
with open('toydataset/captions/captions_train2017.json') as f:
    data = json.load(f)
    for i in range(len(all_file_name)):
        id = int(all_file_name[i].split('.')[0])
        #caps = getCaps.allcaptions(id,data)
        cap = getCaps.onerandomcaption(id,data)
        print(cap)