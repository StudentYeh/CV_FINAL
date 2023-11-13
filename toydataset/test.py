import os
# import toydataset.getCaps as getCaps
import getCapstoy as getCaps
import json
dir_path = './toydataset/images/'
all_file_name = os.listdir(dir_path)
with open('toydataset/captions/captions_train2017.json') as f:
    data = json.load(f)
    for i in range(len(all_file_name)):
        id = int(all_file_name[i].split('.')[0])
        id = 6964
        # caps = getCaps.allcaptions(id,data)
        cap = getCaps.onerandomcaption(id,data)
        print(cap)
        break
