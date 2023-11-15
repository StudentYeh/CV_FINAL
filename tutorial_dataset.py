import json
import cv2
import numpy as np

from torch.utils.data import Dataset

import os
from toydataset.getCapstoy import onerandomcaption
# from annotator.util import resize_image, HWC3

class MyDataset(Dataset):
    def __init__(self):
        self.data = []
        with open('./training/fill50k/prompt.json', 'rt') as f:
            for line in f:
                self.data.append(json.loads(line))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        source_filename = item['source']
        target_filename = item['target']
        prompt = item['prompt']

        source = cv2.imread('./training/fill50k/' + source_filename)
        target = cv2.imread('./training/fill50k/' + target_filename)

        # Do not forget that OpenCV read images in BGR order.
        source = cv2.cvtColor(source, cv2.COLOR_BGR2RGB)
        target = cv2.cvtColor(target, cv2.COLOR_BGR2RGB)

        # Normalize source images to [0, 1].
        source = source.astype(np.float32) / 255.0

        # Normalize target images to [-1, 1].
        target = (target.astype(np.float32) / 127.5) - 1.0

        return dict(jpg=target, txt=prompt, hint=source)


class CocoDataset(Dataset):
    def __init__(self):
        dir_path = './toydataset/images/'
        all_file_name = os.listdir(dir_path)
        self.data = []
        with open('./toydataset/captions/captions_train2017.json') as f:
            data = json.load(f)
            for i in range(len(all_file_name)):
                id = all_file_name[i].split('.')[0]
                cap = onerandomcaption(int(id),data)
                self.data.append(dict(source = f"annotations/{id}.png", \
                                      target = f"images/{id}.jpg", \
                                      prompt = cap))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        source_filename = item['source']
        target_filename = item['target']
        prompt = item['prompt']

        source = cv2.imread('./toydataset/' + source_filename)
        target = cv2.imread('./toydataset/' + target_filename)

        # yeh
        source = cv2.resize(source, (512, 512))
        target = cv2.resize(target, (512, 512))

        # Do not forget that OpenCV read images in BGR order.
        source = cv2.cvtColor(source, cv2.COLOR_BGR2RGB)
        target = cv2.cvtColor(target, cv2.COLOR_BGR2RGB)

        # Normalize source images to [0, 1].
        source = source.astype(np.float32) / 255.0

        # Normalize target images to [-1, 1].
        target = (target.astype(np.float32) / 127.5) - 1.0

        return dict(jpg=target, txt=prompt, hint=source)

