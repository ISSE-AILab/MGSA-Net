import os
from tensorflow.python.client import device_lib
from model import trident_Model
import numpy as np
import pandas as pd
import tensorflow as tf

os.environ["CUDA_VISIBLE_DEVICES"] = "2"
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.5
config.gpu_options.allow_growth = True
sess = tf.Session(config = config)

def train():
    '''
    Preprocessing for dataset
    '''
    # Read  data set (Train data from CSV file)
    csvmaskdata = pd.read_csv('./data/mask_256_train.csv')
    csvimagedata = pd.read_csv('./data/image_256_train.csv')

    csvmaskdata_96 = pd.read_csv('./data/mask_96_train.csv')
    csvimagedata_96 = pd.read_csv('./data/image_96_train.csv')

    csvbounding_box = pd.read_csv('./data/bounding_train_256_96.csv')

    maskdata = csvmaskdata.iloc[:, :].values
    imagedata = csvimagedata.iloc[:, :].values
    maskdata_96 = csvmaskdata_96.iloc[:, :].values
    imagedata_96 = csvimagedata_96.iloc[:, :].values

    bounding_box=csvbounding_box.iloc[:, :].values
    # shuffle imagedata and maskdata together
    perm = np.arange(len(csvimagedata))- 1
    perm = np.delete(perm, 0)
    np.random.shuffle(perm)
    imagedata = imagedata[perm]
    maskdata = maskdata[perm]
    maskdata_96 = maskdata_96[perm] 
    imagedata_96 = imagedata_96[perm] 
    bounding_box=bounding_box[perm]
    triunet = trident_Model(256, 256,96,96, channels=1, costname=("dice coefficient",))
    triunet.train(imagedata, maskdata,imagedata_96,maskdata_96,bounding_box, "changePathoflayer.pd", "./changePathoflayer_log/", 3e-4, 1, 70, 4)

train()