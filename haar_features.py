import cv2
import numpy as np
from integral_image import *

#reading an image
image=cv2.imread("C:\\Users\\tusha\\OneDrive\\Desktop\\og.jpeg")
(h,w,d)=image.shape
print("Image shape: ",(h,w,d))
cv2.imshow("Image",image)
cv2.waitKey(0)

#slicing and cropping face ROI
roi=image[550:690,380:520]
print("ROI shape: ",roi.shape)
cv2.imshow("ROI",roi)
cv2.waitKey(0)

#converting to grayscale
gray_roi=cv2.cvtColor(roi,cv2.COLOR_BGR2GRAY)
cv2.imshow("Gray",gray_roi)
cv2.waitKey(0)
print("Gray ROI shape: ",gray_roi.shape)

def edge_haar_vert(img,gray_img,lenh,lenw,threshold):
    out=img.copy()
    (h,w)=gray_img.shape
    for i in range(0,h-lenh):
        for j in range(0,w-lenw):
            temp=gray_img[i:i+lenh,j:j+lenw]
            temp_int=integral_image(temp)
            bsum1=block_sum(temp_int,(0,0),int(lenw/2),lenh)
            bsum2=block_sum(temp_int,(int(lenw/2),0),int(lenw/2),lenh)
            print("block sums:")
            print(bsum1/((lenh/2)*(lenw/2)))
            print(bsum2/((lenh/2)*(lenw/2)))
            if abs((bsum1/((lenh/2)*(lenw/2)))-(bsum2/((lenh/2)*(lenw/2))))>=threshold:
                cv2.rectangle(out,(i,j),(i+lenh-1,j+lenw-1),(0,250,0),2)
    cv2.imshow("Draw",out)
    cv2.waitKey(0)

def edge_haar_hort(img,gray_img,lenh,lenw,threshold):
    out=img.copy()
    (h,w)=gray_img.shape
    for i in range(0,h-lenh):
        for j in range(0,w-lenw):
            temp=gray_img[i:i+lenh,j:j+lenw]
            temp_int=integral_image(temp)
            bsum1=block_sum(temp_int,(0,0),lenw,int(lenh/2))
            bsum2=block_sum(temp_int,(0,int(lenh/2)),lenw,int(lenh/2))
            print("block sums:")
            print(bsum1/((lenh/2)*(lenw/2)))
            print(bsum2/((lenh/2)*(lenw/2)))
            if abs((bsum1/((lenh/2)*(lenw/2)))-(bsum2/((lenh/2)*(lenw/2))))>=threshold:
                cv2.rectangle(out,(i,j),(i+lenh-1,j+lenw-1),(250,0,0),2)
    cv2.imshow("Draw2",out)
    cv2.waitKey(0)

edge_haar_vert(roi,gray_roi,40,40,255)
edge_haar_hort(roi,gray_roi,40,40,255)
                

        
