import os
import cv2
import tqdm


def file_name_path(file_dir):
    """
    get root path,sub_dirs,all_sub_files
    :param file_dir:
    :return:
    """
    for root, dirs, files in os.walk(file_dir):
        if len(dirs):
        #     print("sub_dirs:", dirs)
            return dirs
def file_name_path_sub_file(file_dir):
    for root, dirs, files in os.walk(file_dir):
        if len(files):
        #     print("sub_dirs:", dirs)
            return files

def save_file2csv(image_file_dir,mask_file_dir ,image_file_name,mask_file_name):
    """
    save file path to csv,this is for segmentation
    :param file_dir:preprocess data path
    :param file_name:output csv name
    :return:
    """
    imageout = open(image_file_name, 'w')
    maskout= open(mask_file_name, 'w')
    
    imageout.writelines("filename" + "\n")
    maskout.writelines("filename" + "\n")

    for sub_path in os.listdir(image_file_dir):
        imagepath=image_file_dir  + sub_path
        maskpath=mask_file_dir  + sub_path
        
        imageout.writelines(imagepath + "\n")
        maskout.writelines(maskpath + "\n")
        

# save_file2csv("/raid/data/xuweixin/LIDC_dataprepare/2DUnet_40/image/", "/raid/data/xuweixin/LIDC_dataprepare/2DUnet_40/mask/" ,"./train_96_image.csv", "./train_96_mask.csv")
def pro_out_index(index_range,index_bounding):
    if(index_range[0]<0):
        index_range[1]=index_range[1]-index_range[0]
        index_range[0]=0
        return index_range
    if(index_range[1]>index_bounding):
        index_range[0]=index_range[0]-(index_range[1]-index_bounding)
        index_range[1]=index_bounding
        return index_range
    return index_range

def crop_image(mask_path,image_path,size,name,out):
    w=size[0]//2
    h=size[1]//2
    groundtruth = cv2.imread(mask_path)[:, :, 0]
    image = cv2.imread(image_path)[:, :, 0]
    h1, w1 = groundtruth.shape
    contours, cnt = cv2.findContours(groundtruth.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 1:
        M = cv2.moments(contours[0])
        try:
            center_x = int(M["m10"] / M["m00"])
            center_y = int(M["m01"] / M["m00"])
            y_range=[center_y-h,center_y+h]
            x_range=[center_x-w,center_x+w]

            y_range=pro_out_index(y_range,h1)
            x_range=pro_out_index(x_range,w1)

            out.write(name+","+str(y_range[0])+","+str(x_range[0])+","+str(y_range[1])+","+str(x_range[1])+"\n")

            mask=groundtruth[ y_range[0]:y_range[1] , x_range[0]:x_range[1]]
            image=image[ y_range[0]:y_range[1] , x_range[0]:x_range[1]]
            cv2.imwrite("./mask_96/"+name+"_mask.png", mask)
            cv2.imwrite("./img_96/"+name+".png", image)
        except:
            print(name)

def get_crop_image_mask():
    mask_path='./mask_256/'
    image_path='./img_256/'
    out=open("./bounding_box_256_96.csv","w")
    out.write("name,y_min,x_min,y_max,x_max"+"\n")
    size=[96,96]
    for sub_path in tqdm.tqdm(os.listdir(image_path)):
        name=sub_path[:-4]
        image=image_path+sub_path
        mask=mask_path+name+"_mask.png"
        crop_image(mask,image,size,name,out)
get_crop_image_mask()