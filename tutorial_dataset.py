import json
from cv2 import cv2
import numpy as np

from torch.utils.data import Dataset
import os
from cocostuff.getCaps import onerandomcaption
from annotator.util import resize_image, HWC3, resize_and_pad

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

## process cocostuff
class CocoDataset(Dataset):
    def __init__(self):
        dir_path = './cocostuff/dataset/images/train2017'
        all_file_name = os.listdir(dir_path)
        self.data = []
        with open('./cocostuff/dataset/captions/annotations/captions_train2017.json') as f:
            data = json.load(f)
            for i in range(len(all_file_name)):
                id = all_file_name[i].split('.')[0]
                cap = onerandomcaption(int(id),data)
                # annotations & images contains train2017 and val2017
                self.data.append(dict(source = f"annotations/train2017/{id}.png", \
                                      target = f"images/train2017/{id}.jpg", \
                                      prompt = cap))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        source_filename = item['source']
        target_filename = item['target']
        prompt = item['prompt']

        source = cv2.imread('./cocostuff/dataset/' + source_filename)
        target = cv2.imread('./cocostuff/dataset/' + target_filename)

        # yeh
        # source = cv2.resize(source, (512, 512))
        # target = cv2.resize(target, (512, 512))
        # joanna
        # resize to make it possible to use different image sizes
        # function from annotator/util.py
        '''
        ### method 1
        ## yet the source, target size are both (512,704,3) for 000000000009
        source = HWC3(source)
        target = HWC3(target)
        target = resize_image(target, 512)
        H, W, C = target.shape
        source = resize_image(source, 512)
        source = cv2.resize(source, (W, H), interpolation=cv2.INTER_LINEAR)
        '''
        '''
        ### method 2
        ## possible to use the resize img from method 1
        # Resize both images to 512x512
        source = cv2.resize(source, (512, 512), interpolation=cv2.INTER_LINEAR)
        target = cv2.resize(target, (512, 512), interpolation=cv2.INTER_LINEAR)
        # Convert to appropriate color space
        source = HWC3(source)
        target = HWC3(target)
        '''
        
        ### method 3
        ## resize and pad to 512x512
        source = resize_and_pad(source, 512)
        target = resize_and_pad(target, 512)
        # Convert to appropriate color space
        source = HWC3(source)
        target = HWC3(target)
        
        # Do not forget that OpenCV read images in BGR order.
        source = cv2.cvtColor(source, cv2.COLOR_BGR2RGB)
        target = cv2.cvtColor(target, cv2.COLOR_BGR2RGB)

        # Normalize source images to [0, 1].
        source = source.astype(np.float32) / 255.0

        # Normalize target images to [-1, 1].
        target = (target.astype(np.float32) / 127.5) - 1.0

        return dict(jpg=target, txt=prompt, hint=source)

