import cv2
import os
import numpy as np

# file paths list for files in a folder
def file_paths(folder_path):
    f_paths=[]
    for file in os.listdir(folder_path):
        eachfile_path=os.path.join(folder_path,file)
        if os.path.isfile(eachfile_path):
            f_paths.append(eachfile_path)
    return f_paths

# getting rois from images
# since not many  non-facial images were available in the used dataset we do data augmentation here to get 'n' rois from one image 
def get_rois(img,newh,neww,n):
    (h,w)=img.shape[:2]
    print(h,w)
    roi_list=[]
    x_stride = int(w /(n))
    y_stride = int(h /(n))
    if n<= (int((h-newh)/y_stride)+1)*(int((w-neww)/x_stride)+1):
        x_starts = np.arange(0, w, x_stride)[:int(np.sqrt(n))+1]
        y_starts = np.arange(0, h, y_stride)[:int(np.sqrt(n))+1]
        print(len(x_starts),len(y_starts))
        for y in y_starts:
            for x in x_starts:
                roi_list.append(img[y:y+newh,x:x+neww])
        return roi_list
    else:
        print("Error: cannot extract {} rois from image shape ({},{}) with spread out strides".format(n,h,w))
        return []
    

# actual code for data augmenting to generate the NON_facial_images folder
if __name__ == "__main__":

    image_paths=file_paths("C:\\Users\\tusha\\Downloads\\iccv09Data.tar\\iccv09Data\\iccv09Data\\images")
    new_dir_path=os.path.join("C:\\Users\\tusha\\OneDrive\\Desktop", "NON_facial_images")
    if not os.path.exists(new_dir_path):
        os.mkdir(new_dir_path)
    print(len(image_paths))
    i=0
    for each in image_paths:
        img=cv2.imread(each)
        roi_list=get_rois(img,128,128,14)
        for j in range(i,i+len(roi_list)):
            cv2.imwrite(new_dir_path+"\\"+str(j)+".png",roi_list[j-i])
        i=j+1
            
    
    print("DONE")
    



