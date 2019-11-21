## Deepsingularity Image Data Augumentation FrameWork for Detection!

### 1. 更新

```version 0.1.0```

第一次提交版本,涵盖数据扩充和标注文件格式转换脚本

### 2. 介绍

#### 2.1 依赖

`PIL`,`opencv`,`skimage`

#### 2.2 数据扩充

#### 2.2.1 功能介绍

数据扩充按照是否改变图像尺寸分为两大类 keep_size 和 change_size,

keep_size支持的数据扩充方式:

1. 图像色彩平衡调节:PIL.ImageEnhance.Color()
2. 图像对比度调节:PIL.ImageEnhance.Contrast()
3. 图像亮度调节:PIL.ImageEnhance.Brightness()
4. 图像加噪声:通过skimage.util.random_noise()实现,支持:高斯噪声、盐/椒噪声、泊松噪声、乘法噪声
5. 图像模糊:PIL.ImageFilter

change_size支持的数据扩充方式:

1. 旋转:选择0-360度范围内的旋转
2. 翻转:水平翻转,垂直翻转
3. 缩放:按照一定比例缩放图片

#### 2.2.2 使用方法

Step 1: 准备 csv 格式的标注文件 `train_labels.csv` ,样例如下
```
/mfs/home/zhuchaojie/ds/data/000.jpg,145,245,324,654,helmet
```
Step 2: 修改 `ds_aug_detection/data_aug/config.py`相关配置,解释如下
```
class DefaultConfigs(object):
    image_files = "/home/user/zcj/detect_frameworks/data/csv_ds/images/" # 原始图像存放路径
    csv_annotations = "/home/user/zcj/detect_frameworks/data/csv_ds/test_aug.csv"  #原始csv格式标注文件存放路径
    raw_data_format = "csv"            # 原始数据格式暂时只支持csv"voc","coco","csv","txt","labelme"
    save_format = "csv"                # annotation saved format                 
    csv_anno_saved = "/home/user/zcj/detect_frameworks/data/csv_ds/csvs/" # 扩充后的csv格式的文件存放路径
    image_format = "jpg"               # original image file format
    image_saved_path = "/home/user/zcj/detect_frameworks/data/csv_ds/auged_images/"# 扩充后的图像保存地址
    classes = ["helmet"]               # all classes

config = DefaultConfigs()
```

Step 3: 执行 ``python example.py``

**温馨提示: 记得将扩充的数据和原始数据合并后再转换格式**

### 2.3 标注格式转换:

说明:由于csv,txt格式过于简单,不提供转换脚本,直接使用 python open就可以完成.

目前支持的格式转换:

- csv to coco2017
- csv to voc2007
- labelme to coco2017
- labelme to voc2007
- txt to coco2017
