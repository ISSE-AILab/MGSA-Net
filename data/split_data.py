import os
import pandas as pd
import tqdm


def get_train():
    im_test = open('./image_train_96.csv', 'w')
    mk_test = open('./mask_train_96.csv', 'w')

    im_test.writelines("filename" + "\n")
    mk_test.writelines("filename" + "\n")

    csvmaskdata = pd.read_csv('./train_96_mask.csv')
    csvimagedata = pd.read_csv('./train_96_image.csv')
    maskdata = csvmaskdata.iloc[:, :].values
    imagedata = csvimagedata.iloc[:, :].values

    for i in range(2000):
        im_test.writelines(imagedata[i][0] + "\n")
        mk_test.writelines(maskdata[i][0] + "\n")

def get_test():
    im_test = open('./image_test_96.csv', 'w')
    mk_test = open('./mask_test_96.csv', 'w')

    im_test.writelines("filename" + "\n")
    mk_test.writelines("filename" + "\n")

    csvmaskdata = pd.read_csv('./train_96_mask.csv')
    csvimagedata = pd.read_csv('./train_96_image.csv')
    maskdata = csvmaskdata.iloc[:, :].values
    imagedata = csvimagedata.iloc[:, :].values

    for i in range(2000,len(maskdata)):
        im_test.writelines(imagedata[i][0] + "\n")
        mk_test.writelines(maskdata[i][0] + "\n")

# get_train()
# get_test()
def find_bounding(name):
    bounding_train=pd.read_csv("./bounding_box_512resize_96.csv")
    bounding_train = bounding_train.iloc[:, :].values
    for i in range(bounding_train.shape[0]):
        if bounding_train[i][0]==name:
            return [bounding_train[i][1],bounding_train[i][2],bounding_train[i][3],bounding_train[i][4]]

def get_bounding_train():
    out=open("./bounding_train_512resize_96.csv","w")
    out.write("name,y_min,x_min,y_max,x_max"+"\n")

    image_train=pd.read_csv("./image_train.csv")
    image_train = image_train.iloc[:, :].values

    for i in tqdm.tqdm(range(image_train.shape[0])):
        sub_list=image_train[i][0]
        name=sub_list.split("/")[-1][:-4]
        bounding_list=find_bounding(name)
        out.write(name+","+str(bounding_list[0])+","+str(bounding_list[1])+","+str(bounding_list[2])+","+str(bounding_list[3])+"\n")

def get_bounding_test():
    out=open("./bounding_test_512resize_96.csv","w")
    out.write("name,y_min,x_min,y_max,x_max"+"\n")

    image_train=pd.read_csv("./image_test.csv")
    image_train = image_train.iloc[:, :].values

    for i in tqdm.tqdm(range(image_train.shape[0])):
        sub_list=image_train[i][0]
        name=sub_list.split("/")[-1][:-4]
        bounding_list=find_bounding(name)
        out.write(name+","+str(bounding_list[0])+","+str(bounding_list[1])+","+str(bounding_list[2])+","+str(bounding_list[3])+"\n")

get_bounding_train()
get_bounding_test()
    

