import os
import cv2
import numpy as np 
import pandas as pd 
from tqdm import tqdm 
from glob import glob
from PIL import Image
from data_aug import *
from convert import *
from config import config
from IPython import embed
all_augumentors = [
		#"color",
		"contrast",
		"brightness",
		#"noise",   #注意！ 数据本身比较暗比较模糊的不要加噪声！
		#"blur",
		"rotate",
                "horizontal_flip",
		#"vertical_flip",
		"scale",
		"enhance",
		"enhance_and_flip",
		]

all_noise_type = ["gaussian","localvar","poisson","salt","pepper","s&p","speckle"]

all_data_format = ["csv",
										#"coco",
										#"txt",
										#"voc"
										]
class Augumentor():
    def __init__(self):
        self.raw_format = config.raw_data_format             # 原始数据格式,以csv格式为例
        self.save_format = config.save_format                # 增强后保存的数据格式
        self.image_paths = glob(config.image_files+"/*.%s"%config.image_format)   # 默认图片格式为jpg
        self.annotations_path = config.csv_annotations
        self.nlc = keep_size()     # 不改变原始图像大小的增强方式
        self.wlc = change_size(config)   # 改变原始图像大小的增强方式
        self.brightness_scalers = [0.8, 1.2]  # factor
        self.contrast_scalers = [0.8, 1.4]  # factor
        self.color_scalers = [0.8, 1.4]  # factor
        self.blur_scalers = [0.8, 1.0, 1.2]  # factor
        self.zoom_scalers = [0.8, 0.9, 1.0]  # factor
        self.angles_scalers = [-10, -5, 5, 10]

    def fit(self):
        assert self.raw_format in all_data_format
        total_boxes = {}
        # read box info for csv format
        if self.raw_format == "csv":
            annotations = pd.read_csv(self.annotations_path,header=None).values
            for annotation in annotations:
                key = annotation[0].split(os.sep)[-1]
                value = np.array([annotation[1:]])
                if key in total_boxes.keys():
                    total_boxes[key] = np.concatenate((total_boxes[key],value),axis=0)
                else:
                    total_boxes[key] = value      
        # read image and process boxes
        for image_path in tqdm(self.image_paths):
            image = Image.open(image_path)
            try:
                raw_boxes = total_boxes[image_path.split(os.sep)[-1]].tolist()  # convert csv box to list
            except KeyError:
                print("%s  must in test list, Or perhaps just no label file"%(image_path))
                continue

            img_file_name = config.image_saved_path+image_path.split(os.sep)[-1].split("."+config.image_format)[0]

            # contrast 
            for scaler in [np.random.choice(self.contrast_scalers)]:
                if "contrast" not in all_augumentors:
                    break
                contrasted_image,contrasted_box = self.nlc.contrast(image,raw_boxes,num=scaler)
                self.write_csv(img_file_name+"_contrasted_%s.%s"%(scaler, config.image_format), contrasted_box)
                self.write_image(contrasted_image,img_file_name+"_contrasted_%s.%s"%(scaler, config.image_format))

            # color banlance
            for scaler in [np.random.choice(self.color_scalers)]:
                if "color" not in all_augumentors:
                    break
                colored_image,colored_box = self.nlc.color(image,raw_boxes,num=scaler)
                self.write_csv(img_file_name+"_colored_%s.%s"%(scaler, config.image_format), colored_box)
                self.write_image(colored_image,img_file_name+"_colored_%s.%s"%(scaler, config.image_format))

            # brightness change
            for scaler in [np.random.choice(self.brightness_scalers)]:
                if "brightness" not in all_augumentors:
                    break
                brightness_image,brightness_box = self.nlc.brightness(image,raw_boxes,num=scaler)
                self.write_csv(img_file_name+"_brightness_%s.%s"%(scaler, config.image_format), brightness_box)
                self.write_image(brightness_image,img_file_name+"_brightness_%s.%s"%(scaler, config.image_format))

            # blur 
            for scaler in [np.random.choice(self.blur_scalers)]:
                if "blur" not in all_augumentors:
                    break
                blured_image,blured_box = self.nlc.blur(image,raw_boxes,filter_type="gaussian",radius=scaler)
                self.write_csv(img_file_name+"_blured_%s.%s"%(scaler, config.image_format), blured_box)
                self.write_image(blured_image,img_file_name+"_blured_%s.%s"%(scaler, config.image_format)) 

            raw_labels = np.array(raw_boxes)[:,-1]
            # scale 
            for scaler in [np.random.choice(self.zoom_scalers)]: 
                if "scale" not in all_augumentors:
                    break
                if scaler<1:
                    scale_image,scale_box = self.wlc.scale(image,np.array(raw_boxes)[:,:-1].tolist(),ratio=[1-scaler, 1-scaler])
                    self.write_csv(img_file_name+"_scale_%s.%s"%(scaler, config.image_format), [raw_labels,scale_box],original=False)
                    self.write_image(scale_image,img_file_name+"_scale_%s.%s"%(scaler, config.image_format))

            # rotate image
            for angle in self.angles_scalers:
                if "rotate" not in all_augumentors:
                    break
                trans_boxes = np.array(raw_boxes)[:,:-1].tolist()   # 传入改变图像大小函数中
                rotated_image,rotated_box = self.wlc.rotate(image,trans_boxes,angle)
                self.write_csv(img_file_name+"_rotated_{}.{}".format(str(angle),config.image_format),[raw_labels,rotated_box],original=False)
                self.write_image(rotated_image,img_file_name+"_rotated_{}.{}".format(str(angle),config.image_format))   

            ## horizontal_flip 
            if "horizontal_flip" not in all_augumentors:
                pass
            else:
                hf_image,hf_box = self.wlc.horizontal_flip(image,np.array(raw_boxes)[:,:-1].tolist())
                self.write_csv(img_file_name+"_hf.%s"%config.image_format,[raw_labels,hf_box],original=False)
                self.write_image(hf_image,img_file_name+"_hf.%s"%config.image_format)   

            #vertical_flip 
            if "vertical_flip" not in all_augumentors:
                pass
            else:
                vf_image,vf_box = self.wlc.vertical_flip(image,np.array(raw_boxes)[:,:-1].tolist())
                self.write_csv(img_file_name+"_vf.%s"%config.image_format,[raw_labels,vf_box],original=False)
                self.write_image(vf_image,img_file_name+"_vf.%s"%config.image_format)         

            #noise change default guassion
            if "noise" not in all_augumentors:
                pass
            else:
                gau_noise_image,gau_noise_box = self.nlc.noise(image,raw_boxes,noise_type="gaussian")
                self.write_csv(img_file_name+"_gau_noise.%s"%(config.image_format), gau_noise_box)
                self.write_image(gau_noise_image,img_file_name+"_gau_noise.%s"%(config.image_format))

            #enhance
            if "enhance" not in all_augumentors:
                pass
            else:
                enhanced_image, enhanced_box = self.nlc.edge_enhance(image,raw_boxes,noise_type="gaussian")
                self.write_csv(img_file_name+"_enhanced_noised.%s"%(config.image_format), enhanced_box)
                self.write_image(enhanced_image,img_file_name+"_enhanced_noised.%s"%(config.image_format))
 
            #enhance and flipx
            if "enhance_and_flip" not in all_augumentors:
                pass
            else:
                enhanced_image, enhanced_box = self.nlc.edge_enhance(image,raw_boxes,noise_type="gaussian")
                e_f_image,e_f_box = self.wlc.horizontal_flip(enhanced_image,np.array(enhanced_box)[:,:-1].tolist())
                self.write_csv(img_file_name+"_enhanced_and_flip.%s"%(config.image_format), [raw_labels,e_f_box], original=False)
                self.write_image(e_f_image,img_file_name+"_enhanced_and_flip.%s"%(config.image_format))

    def write_csv(self,filename,boxes,original=True):
        saved_file = open(config.csv_anno_saved+"/augmented_boxes.csv","a+")
        if original:
            new_boxes = boxes
            for new_box in new_boxes:
                label = new_box[-1]
                saved_file.write(filename+","+str(new_box[0])+","+str(new_box[1])+","+str(new_box[2])+","+str(new_box[3]) + ","+label+"\n")
        else:
            labels, new_boxes= boxes[0],boxes[1]
            for label,new_box in zip(labels,new_boxes):
                saved_file.write(filename+","+str(new_box[0])+","+str(new_box[1])+","+str(new_box[2])+","+str(new_box[3]) + ","+label+"\n")

    def write_image(self,image,filename):
        image.save(filename) 

if __name__ == "__main__":
    detection_augumentor = Augumentor()
    detection_augumentor.fit()
