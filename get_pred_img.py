import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from tensorflow.python.client import device_lib

import cv2
from model import trident_Model
import numpy as np
import pandas as pd
import tqdm

trident_unet = trident_Model(256, 256,96,96, channels=1, costname=("dice coefficient",), inference=True,
                          model_path="./crop_d2_log/model/crop_d2.pd-62873")
def dice_sorce(y_true,y_pred):

    smooth = 1.
    intersection = np.sum(y_true * y_pred)
    union = np.sum(y_true + y_pred)
    numerator = 2.0 * intersection+smooth
    denominator = union+smooth
    coef = numerator / denominator
    return coef

def predict(image_path,mask_path,image_128_path,mask_128_path,bbox,out):

    name=image_path.split("/")[-1][:-4]

    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    
    img_128 = cv2.imread(image_128_path, cv2.IMREAD_GRAYSCALE)
    mask_128 = cv2.imread(mask_128_path, cv2.IMREAD_GRAYSCALE)

    dice_mask=mask/255.
    dice_128_mask=mask_128/255.

    test_imges=np.array(img)[np.newaxis,:,:,np.newaxis]
    test_masks=np.array(mask)[np.newaxis,:,:,np.newaxis]
    test_128_imges=np.array(img_128)[np.newaxis,:,:,np.newaxis]
    test_128_masks=np.array(mask_128)[np.newaxis,:,:,np.newaxis]
    bbox=np.array(bbox)
    
    pred_lsit,pred_128 = trident_unet.prediction(test_imges,test_masks,test_128_imges,test_128_masks,bbox)

    # cv2.imwrite('./pred_25696/bran1/'+name+'.png',pred_lsit[0])
    # cv2.imwrite('./pred_25696/bran2/'+name+'.png',pred_lsit[1])
    # cv2.imwrite('./pred_25696/bran3/'+name+'.png',pred_lsit[2])

    # cv2.imwrite('./pred_25696/local_p/'+name+'.png',pred_128)

    dice_pred=pred_lsit[0]/255.
    dice1=dice_sorce(dice_mask,dice_pred)
    dice_pred=pred_lsit[1]/255.
    dice2=dice_sorce(dice_mask,dice_pred)
    dice_pred=pred_lsit[2]/255.
    dice3=dice_sorce(dice_mask,dice_pred)
    dice_pred=pred_128/255.
    dice_local=dice_sorce(dice_128_mask,dice_pred)
    out.write(name+","+str(dice1)+","+str(dice2)+","+str(dice3)+","+str(dice_local)+"\n")

csvmaskdata = pd.read_csv('data/mask_256_test.csv')
csvimagedata = pd.read_csv('data/image_256_test.csv')
csvmask_128_data = pd.read_csv('data/mask_96_test.csv')
csvimage_128_data = pd.read_csv('data/image_96_test.csv')
csvbboxdata = pd.read_csv('data/bounding_test_256_96.csv')

maskdata = csvmaskdata.iloc[:, :].values
imagedata = csvimagedata.iloc[:, :].values
mask_128_data = csvmask_128_data.iloc[:, :].values
image_128_data = csvimage_128_data.iloc[:, :].values
bbox_data = csvbboxdata.iloc[:, :].values

assert len(maskdata)==len(imagedata) and len(imagedata)==len(mask_128_data) and len(mask_128_data) == len(image_128_data) and len(image_128_data)== len(bbox_data)
out=open("./changePath.csv","w")
out.write("filename,branch1,branch2,branch3,local"+"\n")

for i in tqdm.tqdm(range(len(imagedata))):
    image_path=imagedata[i][0]
    mask_path=maskdata[i][0]
    image_128_path=image_128_data[i][0]
    mask_128_path=mask_128_data[i][0]
    bbox=[[bbox_data[i][1],bbox_data[i][2],bbox_data[i][3],bbox_data[i][4]]]
    predict(image_path,mask_path,image_128_path,mask_128_path,bbox,out)