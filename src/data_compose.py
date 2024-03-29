import glob 
import numpy as np
import matplotlib.pyplot as plt
import os,sys,time
import pandas as pd
from IPython import embed
import cv2
from matplotlib.patches import Circle
import csv2labelme
import labelme2coco

def get_data(data_path, individual_resolution_height, individual_resolution_width):
    data = cv2.imread(data_path)
    data = cv2.resize(data, (individual_resolution_width, individual_resolution_height))
    return data

def get_affine(data):
    randomer1 = 1+0.05*np.random.randn()
    randomer2 = 1+0.05*np.random.randn()
    randomer3 = 1+0.05*np.random.randn()
    randomer4 = 1+0.05*np.random.randn()
    randomer5 = 1+0.05*np.random.randn()
    randomer6 = 1+0.05*np.random.randn()
    pts1 = np.float32([[width/10,height/10],[width*9/10,height*9/10],[width/10,height*9/10]])
    pts2 = np.float32([[randomer1*width/10,randomer2*height/10],[randomer3*width*9/10,randomer4*height*9/10],[randomer5*width/10,randomer6*height*9/10]])
    M = cv2.getAffineTransform(pts1,pts2)
    data = cv2.warpAffine(data, M, (width, height))
    return data

def get_perspective(data, fac=0.01):
    randomer1 = 1+fac*np.random.randn()
    randomer2 = 1+fac*np.random.randn()
    randomer3 = 1+fac*np.random.randn()
    randomer4 = 1+fac*np.random.randn()
    randomer5 = 1+fac*np.random.randn()
    randomer6 = 1+fac*np.random.randn()
    randomer7 = 1+fac*np.random.randn()
    randomer8 = 1+fac*np.random.randn()
    height, width, channel  = data.shape
    pts1 = np.float32([[width/10,height/10],[width*9/10,height*9/10],[width/10,height*9/10],[width*9/10,height/10]])
    pts2 = np.float32([[randomer1*width/10,randomer2*height/10],[randomer3*width*9/10,randomer4*height*9/10],[randomer5*width/10,randomer6*height*9/10],[randomer7*width*9/10,randomer8*height/10]])
    M = cv2.getPerspectiveTransform(pts1,pts2)
    data = cv2.warpPerspective(data, M, (width, height))
    return data

def get_direction_rotate(data):
    randomer = np.random.randint(4)
    if data.shape[0]!=data.shape[1]:randomer*=2    #如果长宽不等则不能转1/4圈
    height, width, channel  = data.shape
    M = cv2.getRotationMatrix2D(((width-1)/2.0,(height-1)/2.0),90*randomer,1)
    data = cv2.warpAffine(data,M,(width,height))
    return data

def get_turb_rotate(data, fac=1):
    data = get_direction_rotate(data)
    randomer = 0+fac*np.random.randn()
    height, width, channel  = data.shape
    M = cv2.getRotationMatrix2D(((width-1)/2.0,(height-1)/2.0), randomer, 1)
    data = cv2.warpAffine(data,M,(width,height))
    return data

def get_translation(data, fac=2):
    height, width, channel  = data.shape
    randomer1 = (fac*np.random.randn())*np.random.choice([1,-1])
    randomer2 = (fac*np.random.randn())*np.random.choice([1,-1])
    M = np.float32([[1,0,randomer1],[0,1,randomer2]])
    data = cv2.warpAffine(data,M,(width,height))
    return data

def get_filter(data, size=5):
    kernel = np.ones((size,size),np.float32)/(size**2)
    data = cv2.filter2D(data, -1, kernel)
    return data

def get_bifilter(data):
    diameter = np.random.randint(10)
    color = np.random.randint(20)
    spatial = np.random.randint(10)
    #Blur while keep boundary
    #???? bifilter together with binary will given new vein
    data = cv2.bilateralFilter(data, diameter,color, spatial)
    return data

def get_open_close(data):
    size = np.random.randint(10)
    kernel = np.ones((size,size),np.uint8)
    data = cv2.morphologyEx(data, cv2.MORPH_OPEN, kernel)
    data = cv2.morphologyEx(data, cv2.MORPH_CLOSE, kernel)
    #data = cv2.dilate(data, kernel,iterations = 1)
    #data = cv2.erode(data, kernel, iterations = 5)
    return data

def get_adaptive_binary(data):
    height, width, channel  = data.shape
    block_size = int(2*int(width/200)+1)
    #TODO: size should depend on physical size, then it will blender AND retain details of block.
    #Very big block: almost split whole pile.
    #Very small block: keeps every noise details.
    constant = 1
    gray_data = cv2.cvtColor(data, cv2.COLOR_BGR2GRAY)
    data = cv2.adaptiveThreshold(gray_data, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, block_size, constant)
    return gray_data, data

def get_gradient(data):
    data = np.abs(cv2.Laplacian(data, cv2.CV_64F))
    data = np.clip(data, 0, 255)
    data = np.uint8(data)
    return data

def get_denoise(data):
    size = np.random.randint(10)
    data = cv2.fastNlMeansDenoisingColored(data,None,size,1,7,7)
    return data

def cart2pol(x, y):
    radius = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return(radius, phi)

def pol2cart(radius, phi):
    x = radius * np.cos(phi)
    y = radius * np.sin(phi)
    return(x, y)


def get_brightness(src1):
    randomer = np.random.randint(80, 100)/100
    height, width, channel  = src1.shape
    h, w, ch = src1.shape#获取shape的数值，height和width、通道
    #新建全零图片数组src2,将height和width，类型设置为原图片的通道类型(色素全为零，输出为全黑图片)
    src2 = np.zeros([h, w, ch], src1.dtype)
    dst = cv2.addWeighted(src1, randomer, src2, 1-randomer, 1)
    return dst

def get_contrast(src1):
    randomer = np.random.randint(-50, 30)
    height, width, channel  = src1.shape
    h, w, ch = src1.shape#获取shape的数值，height和width、通道
    #新建全零图片数组src2,将height和width，类型设置为原图片的通道类型(色素全为零，输出为全黑图片)
    src2 = np.zeros([h, w, ch], src1.dtype)
    dst = cv2.addWeighted(src1, 1, src2, 0, randomer)
    return dst

def get_backlight(data, individual=True):    #Serves as vein render for individual
    status = True
    fac = 2 if individual else 10   #vein
    height, width, channel  = data.shape
    x = np.arange(0, height)
    y = np.arange(0, width)
    X, Y = np.meshgrid(x, y)
    if individual:
        randomer1 = 1/(10**np.random.randint(4, 7))
        randomer2 = 1/(10**np.random.randint(0, 20))
        #randomer1 = 1e-4
        #randomer2 = 1e-6
    else:
        randomer1 = 1/(10**np.random.randint(7, 10))
        randomer2 = 1/(10**np.random.randint(7, 10))
 #      randomer1 = 1/(10**np.random.randint(9, 11))
  #      randomer2 = 1/(10**np.random.randint(8, 11))
    randomer3 = np.random.randint(0,1000)/100
    randomer4 = np.random.randint(-2*fac,fac)/10
    background = np.sin(randomer3+np.sqrt(randomer1*X ** 3 + randomer2*Y ** 3))+1
    background = (background/background.max()*255).T
    background = np.clip(background, 0, 255)
    background = np.tile(background, (3,1,1)).transpose(1,2,0)
    tmp_data = np.uint8(np.clip(data+randomer4*background, 0, 255))
    while 1:
        too_low = len(np.where(tmp_data.sum(axis=2)<30*3)[0])/(height*width)
        too_high = len(np.where(tmp_data.sum(axis=2)>225*3)[0])/(height*width)
        if too_low > 0.10 or too_high>0.1:
            randomer4 *= 0.9
            tmp_data = np.uint8(np.clip(data+randomer4*background, 0, 255))
            #print("Reducing factor...", randomer4, "(",too_low, too_high, ")")
            if np.abs(randomer4)<1e-2:
                status = False
                break
        else:
            break
    return tmp_data, status

###############################################################################################

def get_ADR(data):   #Automatic domain randomization
    #morphology:
    data = get_open_close(data)
    data = get_bifilter(data)
    data = get_contrast(data)
    data = get_brightness(data)
    data, _ = get_backlight(data)

    #Seq matters:
    #Geometry:
    data = get_translation(data)
    data = get_perspective(data)
    data = get_turb_rotate(data)

    return data

def get_json(num_in_col, num_in_row, individual_resolution_width, individual_resolution_height, figname):
    anno_noise = 0.05
    json_width = np.arange(int(individual_resolution_width/2), num_in_col*individual_resolution_width, individual_resolution_width)
    json_height = np.arange(int(individual_resolution_height/2), num_in_row*individual_resolution_height, individual_resolution_height)
    #center to boundary
    top_x = json_width-individual_resolution_width/2
    top_y = json_height-individual_resolution_height/2
    bottom_x = json_width+individual_resolution_width/2
    bottom_y = json_height+individual_resolution_height/2
    #meshgrid
    json_X1, json_Y1 = np.meshgrid(top_x, top_y)
    json_X2, json_Y2 = np.meshgrid(bottom_x, bottom_y)
    json_X1, json_Y1, json_X2, json_Y2 = json_X1.flatten(), json_Y1.flatten(), json_X2.flatten(), json_Y2.flatten()
    #Add salt:
    json_X1 = np.clip(json_X1 - np.abs(anno_noise*individual_resolution_width*np.random.randn(len(json_X1))), 0, num_in_col*individual_resolution_width)
    json_Y1 = np.clip(json_Y1 - np.abs(anno_noise*individual_resolution_height*np.random.randn(len(json_Y1))), 0, num_in_row*individual_resolution_height)
    json_X2 = np.clip(json_X2 + np.abs(anno_noise*individual_resolution_width*np.random.randn(len(json_X2))), 0, num_in_col*individual_resolution_width)
    json_Y2 = np.clip(json_Y2 + np.abs(anno_noise*individual_resolution_height*np.random.randn(len(json_Y2))), 0, num_in_row*individual_resolution_height)

    content = np.round(np.vstack((json_X1, json_Y1, json_X2, json_Y2)).T)
    jsons = pd.DataFrame(content.astype(int), columns=[1,2,3,4])
    jsons['path'] = figname+".jpg"
    jsons['class'] = "brick"
    jsons = jsons[['path',1,2,3,4, "class"]]
    jsons.to_csv("../data_generation/%s.csv"%figname, header=None, index=None)
    print("Saving ../data_generation/%s.json"%figname)
    return json_X1, json_Y1, json_X2, json_Y2    

def generate_img_and_json(num_in_row, num_in_col, data_cad, individual_resolution_height, individual_resolution_width):
    height, width, channel  = data_cad[0].shape
    #Image data:
    base = np.empty(shape=(height, 0, 3))
    for i in range(num_in_row*num_in_col):
        which = np.random.randint(len(data_name))
        ADR = get_ADR(data_cad[which])       #Automatic domain randomization
        base = np.hstack((base, ADR))   
    base = np.uint8(base)
    base = base.reshape(num_in_row*height, num_in_col*width, 3)

    #complicate rearrange....
    index = np.array([])
    tmp = np.arange(0, height*num_in_row, height)
    for _ in range(height):
        index = np.append(index, tmp)
        tmp += 1
    index = index.astype(int)
    base = base[index.argsort()]

    #base = get_perspective(base, 0.05)
    base, status = get_backlight(base, individual = False)

    plt.imshow(base[:,:,[2,1,0]])
    num = str(len(glob.glob("../data_generation/*.jpg"))).zfill(5)
    figname = "generate_%s"%num
    cv2.imwrite("../data_generation/%s.jpg"%figname, base)
    print("Saving ../data_generation/%s.jpg"%figname)

    #Get Json:
    json_X1, json_Y1, json_X2, json_Y2 = get_json(num_in_col, num_in_row, individual_resolution_width, individual_resolution_height, figname)
    plt.scatter(json_X1, json_Y1)
    plt.scatter(json_X2, json_Y2)
    #plt.show()
    #embed()
    return base, figname


if __name__ == "__main__":
    print("start...")

    data_cad = {}
    data_name = ["1_1.jpg", "2_2.jpg","1.jpg", "2.jpg"]
    #data_name = ["1.jpg"]
    individual_resolution_height = 100
    individual_resolution_width = 100
    for idx,i in enumerate(data_name):
        data_cad[idx] = get_data('../cad/'+i, individual_resolution_height=individual_resolution_height, individual_resolution_width=individual_resolution_width)
    distance_cad = 0.1   #m
    
    #Generate!
    num_in_col = 12
    num_in_row = 18
    
    for i in range(1):
        base,figname = generate_img_and_json(num_in_row, num_in_col, data_cad, individual_resolution_height, individual_resolution_width)
	#Convert each csv to each labelme json:
        csv2labelme.single_csv_to_labelme_json("../data_generation/", "../data_generation/%s.csv"%figname)
    #Convert all labelme jsons to a coco json:
    runner = labelme2coco.labelme2coco(glob.glob('../data_generation/*.json'), 'output_coco.json', None, [], {'brick':1})
    runner.save_json()

    #embed()




