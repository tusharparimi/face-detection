import cv2
import os
import numpy as np
from nonfacial_data_augmentation import file_paths



if __name__ == "__main__":

    image_paths=file_paths("C:\\Users\\tusha\\Downloads\\thumbnails128x128\\thumbnails128x128")
    new_dir_path=os.path.join("C:\\Users\\tusha\\OneDrive\\Desktop", "facial_images")
    if not os.path.exists(new_dir_path):
        os.mkdir(new_dir_path)
    print(len(image_paths))
    for i in range(0,11440):
        img=cv2.imread(image_paths[i])
        if img.shape[:2]== (128,128):
            cv2.imwrite(new_dir_path+"\\"+str(i)+".png",img)
    print("DONE")
