class DefaultConfigs(object):
    image_files = "/mfs/home/limengwei/NanKang_WenHuaJiaRui/data/train_det/"    # your data file
    raw_data_format = "csv"            # "voc","coco","csv","txt","labelme"

    save_format = "csv"                # annotation saved format                 
    csv_anno_saved = "/mfs/home/limengwei/NanKang_WenHuaJiaRui/data/"    # your  data root
    image_format = "jpg"               # original image file format
    csv_annotations = "/mfs/home/limengwei/NanKang_WenHuaJiaRui/data/train_det.csv"    # original annotation file  for csv format
    image_saved_path = "/mfs/home/limengwei/NanKang_WenHuaJiaRui/data/"   # image save path for augumented images
    #classes = ["angle,","angle_r","top","top_r","head"]               # all classes

config = DefaultConfigs()
