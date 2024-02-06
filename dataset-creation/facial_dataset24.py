#This file was runned on google colab


import cv2
import numpy as np
from csv import DictWriter
from csv import writer
import os
from nonfacial_data_augmentation import file_paths
import sys
sys.path.insert(1,'C:\\Users\\tusha\\OneDrive\\Documents\\Projects\\face-detection\\haar-features')
from haar_feature_defs import rect2_haar_hort, rect2_haar_vert, rect3_haar_hort, rect4_haar_diag

def create_csvfile(dict_data,csvfile_name):
    if os.path.exists(csvfile_name):
        with open(csvfile_name,'a') as f_object:
            dictwriter_object=DictWriter(f_object,fieldnames=[col for col in dict_data])
            dictwriter_object.writerow(dict_data)
            f_object.close()
    else:
        with open(csvfile_name,'w') as f_object:
            writer_object=writer(f_object)
            writer_object.writerow([col for col in dict_data])
            writer_object.writerow([dict_data[col] for col in dict_data])
            f_object.close()

if __name__ == "__main__":
    
    facials=file_paths("C:\\Users\\tusha\\OneDrive\\Desktop\\facial_images")
    
    for each in facials:
        img=cv2.imread(each)
        gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        gray24=cv2.resize(gray, (24, 24),interpolation = cv2.INTER_AREA)
        (h24,w24)=gray24.shape
        features={}
        for i in range(1,h24+1):
            for j in range(2,w24+1):
                features.update(rect2_haar_hort(gray24,i,j,i,j))
        
        for i in range(2,h24+1):
            for j in range(1,w24+1):
                features.update(rect2_haar_vert(gray24,i,j,i,j))
        
        for i in range(1,h24+1):
            for j in range(3,w24+1):
                features.update(rect3_haar_hort(gray24,i,j,i,j))
        
        for i in range(2,h24+1):
            for j in range(2,w24+1):
                features.update(rect4_haar_diag(gray24,i,j,i,j))

        create_csvfile(features,"reduced_features_facial_data24_test.csv")
        
        print(each)
        
        