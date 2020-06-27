# coding=utf-8
# =================================================================================================
# DATA:2020/04/20 Mon
# AUTHER:zhangxiangbo
# DESCRIPTION:得到test数据，返回给训练使用
# =================================================================================================
import numpy as np
import pandas as pd
import os
import cv2
import tqdm
import h5py
# import tensorflow as tf
csvmaskdata = pd.read_csv('data/mask_test.csv')
csvimagedata = pd.read_csv('data/image_test.csv')

csvmask_96_data = pd.read_csv('data/mask_96_test.csv')
csvimage_96_data = pd.read_csv('data/image_96_test.csv')
csvbboxdata = pd.read_csv('data/bounding_test_512resize_96.csv')

maskdata = csvmaskdata.iloc[:, :].values
imagedata = csvimagedata.iloc[:, :].values
mask_96_data = csvmask_96_data.iloc[:, :].values
image_96_data = csvimage_96_data.iloc[:, :].values
bbox_data = csvbboxdata.iloc[:, :].values

img_256_h5=[]
mask_256_h5=[]
img_96_h5=[]
mask_96_h5=[]
bbox_h5=[]
for i in tqdm.tqdm(range(len(imagedata))):
    image_path=imagedata[i][0]
    mask_path=maskdata[i][0]
    image_local_path=image_96_data[i][0]
    mask_local_path=mask_96_data[i][0]
    bbox=[[bbox_data[i][1],bbox_data[i][2],bbox_data[i][3],bbox_data[i][4]]]
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    size = (int(img.shape[0]*0.5), int(img.shape[1]*0.5))  
    img = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
    mask = cv2.resize(mask, size, interpolation=cv2.INTER_AREA)

    img_local = cv2.imread(image_local_path, cv2.IMREAD_GRAYSCALE)
    mask_local = cv2.imread(mask_local_path, cv2.IMREAD_GRAYSCALE)
    img_256_h5.append(img)
    mask_256_h5.append(mask)
    img_96_h5.append(img_local)
    mask_96_h5.append(mask_local)
    bbox_h5.append(bbox)

img_256_h5=np.array(img_256_h5)[:,:,:,np.newaxis]
mask_256_h5=np.array(mask_256_h5)[:,:,:,np.newaxis]
img_96_h5=np.array(img_96_h5)[:,:,:,np.newaxis]
mask_96_h5=np.array(mask_96_h5)[:,:,:,np.newaxis]
print(img_256_h5.shape)
print(mask_256_h5.shape)
print(img_96_h5.shape)
print(mask_96_h5.shape)

with h5py.File("./512_resize_96_test.h5","w") as f:
    f["X_512"]=img_256_h5
    f["Y_512"]=mask_256_h5
    f["X_96"]=img_96_h5
    f["Y_96"]=mask_96_h5
    f["bbox"]=bbox_h5