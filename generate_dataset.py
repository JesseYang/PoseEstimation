import sys
sys.path.insert(1, '../data/coco/PythonAPI/')
sys.path.insert(1, '../multi/')
from pycocotools.coco import COCO
from utils.preprocess import anno_to_ours
import random

coco = COCO('data/coco/annotations/person_keypoints_train2014.json')
records_train = []
for img_id in coco.imgs.keys():
    ann_ids = coco.getAnnIds(imgIds = img_id)
    img_dict = coco.imgs[img_id]
    img_anns = coco.loadAnns(ann_ids)
    persons = anno_to_ours(img_anns)
    
    if len(persons) > 0:
        records_train.append('person_keypoints_train2014.json,{}'.format(img_id))

print("The total sample number in training set is %d" % len(records_train))

coco = COCO('data/coco/annotations/person_keypoints_val2014.json')
records_val = []
for img_id in coco.imgs.keys():
    ann_ids = coco.getAnnIds(imgIds = img_id)
    img_dict = coco.imgs[img_id]
    img_anns = coco.loadAnns(ann_ids)
    persons = anno_to_ours(img_anns)
    
    if len(persons) > 0:
        records_val.append('person_keypoints_val2014.json,{}'.format(img_id))

print("The total sample number in validation set is %d" % len(records_val))

records = records_train + records_val

print("The training set and validation set are combined together and re-splitted")

random.shuffle(records)
test_ratio = 0.1
# split into training set and test set
total_num = len(records)
test_num = int(test_ratio * total_num)
train_num = total_num - test_num
train_records = records[0:train_num]
test_records = records[train_num:]
print("train: ", len(train_records))
print("test: ", len(test_records))

with open('data_all.txt', "w") as f:
    f.writelines((i+'\n' for i in records))
with open('data_train.txt', "w") as f:
    f.writelines((i+'\n' for i in train_records))
with open('data_test.txt', "w") as f:
    f.writelines((i+'\n' for i in test_records))

