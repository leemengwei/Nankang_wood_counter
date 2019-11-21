# -*- coding: utf-8 -*-
'''
@time: 2019/01/11 11:28
spytensor
'''

import os
import json
import numpy as np
import pandas as pd
import glob
import cv2
import os
import shutil
# from IPython import embed
from sklearn.model_selection import train_test_split
np.random.seed(41)

#0为背景


class Csv2CoCo:

    def __init__(self,image_dir,total_annos):
        self.images = []
        self.annotations = []
        self.categories = []
        self.img_id = 0
        self.ann_id = 0
        self.image_dir = image_dir
        self.total_annos = total_annos

    def save_coco_json(self, instance, save_path):
        json.dump(instance, open(save_path, 'w'), ensure_ascii=False, indent=2)  # indent=2 更加美观显示

    # 由txt文件构建COCO
    def to_coco(self, keys):
        self._init_categories()
        for key in keys:
            self.images.append(self._image(key))
            shapes = self.total_annos[key]
            
            for i in range(len(keys)):
                bboxi = []
                for cor in shapes[i][0].split(" "):
                    bboxi.append(int(cor))
                annotation = self._annotation(bboxi)
                self.annotations.append(annotation)
                self.ann_id += 1
            self.img_id += 1
        instance = {}
        instance['info'] = 'spytensor created'
        instance['license'] = ['license']
        instance['images'] = self.images
        instance['annotations'] = self.annotations
        instance['categories'] = self.categories
        return instance

    # 构建类别
    def _init_categories(self):
        category = {}
        category['id'] = 0
        category['name'] = 'i'
        self.categories.append(category)

    # 构建COCO的image字段
    def _image(self, path):
        image = {}
        print(path)
        img = cv2.imread(self.image_dir + path)
        image['height'] = img.shape[0]
        image['width'] = img.shape[1]
        image['id'] = self.img_id
        image['file_name'] = path
        return image

    # 构建COCO的annotation字段
    def _annotation(self, shape):
        # label = shape[-1]
        points = shape[:-1]
        annotation = {}
        annotation['id'] = self.ann_id
        annotation['image_id'] = self.img_id
        annotation['category_id'] = 0
        annotation['segmentation'] = self._get_seg(points)
        annotation['bbox'] = self._get_box(points)
        annotation['iscrowd'] = 0
        annotation['area'] = 1.0
        return annotation

    # COCO的格式： [x1,y1,w,h] 对应COCO的bbox格式
    def _get_box(self, points):
        min_x = points[0]
        min_y = points[1]
        width = points[2]
        height = points[3]
        return [min_x, min_y, width, height]
    
    # COCO的格式： [x1,y1,x2,y2...] 对应COCO的seg格式
    def _get_seg(self, points):
        min_x = points[0]
        min_y = points[1]
        width = points[2]
        height = points[3]
        a = []
        a.append([min_x, min_y, min_x+width, min_y, min_x+width, min_y+height, min_x, min_y+height])
        return a

if __name__ == '__main__':
    csv_file = "submit/train_merge_labels.csv"
    image_dir = "submit/augument/"
    # 整合csv格式标注文件
    total_csv_annotations = {}
    annotations = pd.read_csv(csv_file,header=None).values
    for annotation in annotations:
        key = annotation[0].split(os.sep)[-1]
        value = np.array([annotation[1:]])
        if key in total_csv_annotations.keys():
            total_csv_annotations[key] = np.concatenate((total_csv_annotations[key],value),axis=0)
        else:
            total_csv_annotations[key] = value
    # 按照键值划分数据
    total_keys = list(total_csv_annotations.keys())
    train_keys, val_keys = train_test_split(total_keys, test_size=0.2)
    print("train_n:", len(train_keys), 'val_n:', len(val_keys))
    # 创建必须的文件夹
    if not os.path.exists('./coco/annotations/'):
        os.makedirs('./coco/annotations/')
    if not os.path.exists('./coco/images/train2017/'):
        os.makedirs('./coco/images/train2017/')
    if not os.path.exists('./coco/images/val2017/'):
        os.makedirs('./coco/images/val2017/')
    # 把训练集转化为COCO的json格式
    l2c_train = Csv2CoCo(image_dir=image_dir,total_annos=total_csv_annotations)
    train_instance = l2c_train.to_coco(train_keys)
    l2c_train.save_coco_json(train_instance, './coco/annotations/instances_train2017.json')
    for file in train_keys:
        shutil.copy(image_dir+file,"./coco/images/train2017/")
    for file in val_keys:
        shutil.copy(image_dir+file,"./coco/images/val2017/")
    # 把验证集转化为COCO的json格式
    l2c_val = Csv2CoCo(image_dir=image_dir,total_annos=total_csv_annotations)
    val_instance = l2c_val.to_coco(val_keys)
    l2c_val.save_coco_json(val_instance, './coco/annotations/instances_val2017.json')
