import pandas as pd
import os

csvimagedata = pd.read_csv('./image_train.csv')
imagedata = csvimagedata.iloc[:, :].values
for i in range(imagedata.shape[0]):
    
    data_str=''.join(imagedata[i])
    if os.path.exists(data_str):
        print(data_str)
    else :
        print(data_str)