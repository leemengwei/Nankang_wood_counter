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
np.random.seed(41)

#0为背景


class Txt2CoCo:

    def __init__(self,image_dir):
        self.images = []
        self.annotations = []
        self.categories = []
        self.img_id = 0
        self.ann_id = 0
        self.image_dir = image_dir

    def save_coco_json(self, instance, save_path):
        json.dump(instance, open(save_path, 'w'), ensure_ascii=False, indent=2)  # indent=2 更加美观显示

    # 由txt文件构建COCO
    def to_coco(self, txt_path_list):
        self._init_categories()
        
        for txt_path in txt_path_list:
            obj = self.read_txtfile(txt_path)
            self.images.append(self._image(obj, txt_path))
            shapes = obj
            if shapes.ndim < 2:
                shapes = np.array([shapes])
            for i in range(len(shapes[:,0])):
                annotation = self._annotation(shapes[i])
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
    def _image(self, obj, path):
        image = {}
        print(path)
        #path = "/home/user/zcj/gangjin/data/train/"
        img = cv2.imread(self.image_dir + path[:-4].split("/")[-1]+'.jpg')
        # from labelme import utils
        # utils.img_b64_to_arr读取文件很耗时,若图片大小都一致，可以固定h,w
        # img_x = utils.img_b64_to_arr(obj['imageData'])
        # h, w = img_x.shape[:-1]
        image['height'] = img.shape[0]
        image['width'] = img.shape[1]
        image['id'] = self.img_id
        image['file_name'] = os.path.basename(path).replace(".txt", ".jpg")
        return image

    # 构建COCO的annotation字段
    def _annotation(self, shape):
        label = shape[0]
        points = shape[1:5]
        
        annotation = {}
        annotation['id'] = self.ann_id
        annotation['image_id'] = self.img_id
        annotation['category_id'] = 0
        annotation['segmentation'] = self._get_seg(points)
        annotation['bbox'] = self._get_box(points)
        annotation['iscrowd'] = 0
        annotation['area'] = 1.0
        return annotation

    # 读取txt文件，返回一个 array 对象
    def read_txtfile(self, path):
        return np.loadtxt(path)

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


def train_test_split(data, test_size=0.12):
    n_val = int(len(data) * test_size)
    np.random.shuffle(data)
    train_data = data[:-n_val]
    val_data = data[-n_val:]
    return train_data, val_data

if __name__ == '__main__':
    import shutil
    base_dir = 'aug_txts/'
    image_dir = "aug_images/"
    # 获取images目录下所有的txt文件列表
    txt_list_path = glob.glob(base_dir + "/*.txt")
    # 数据划分,这里没有区分val2017和tran2017目录，所有图片都放在images目录下
    train_path, val_path = train_test_split(txt_list_path, test_size=0.2)
    print("train_n:", len(train_path), 'val_n:', len(val_path))
    # train_path = txt_list_path
    # val_path = txt_list_path
    if not os.path.exists('./coco/annotations/'):
        os.mkdir('./coco/annotations/')
    if not os.path.exists('./coco/images/train2017/'):
        os.makedirs('./coco/images/train2017/')
    if not os.path.exists('./coco/images/val2017/'):
        os.makedirs('./coco/images/val2017/')
    # 把训练集转化为COCO的json格式
    l2c_train = Txt2CoCo(image_dir=image_dir)
    train_instance = l2c_train.to_coco(train_path)
    l2c_train.save_coco_json(train_instance, './coco/annotations/instances_train2017.json')
    for file in train_path:
        shutil.copy(file.replace(".txt",".jpg").replace("aug_txts","aug_images"),"./coco/images/train2017/")
    for file in val_path:
        shutil.copy(file.replace(".txt",".jpg").replace("aug_txts","aug_images"),"./coco/images/val2017/")
    # 把验证集转化为COCO的json格式
    l2c_val = Txt2CoCo(image_dir=image_dir)
    val_instance = l2c_val.to_coco(val_path)
    l2c_val.save_coco_json(val_instance, './coco/annotations/instances_val2017.json')
