import cv2
import numpy as np
import sys
sys.path.insert(1,'C:\\Users\\tusha\\OneDrive\\Documents\\Projects\\face-detection\\facial-features')
from integral_image import integral_image, block_sum

def rect2_haar_hort(gray_img,lenh,lenw,stride_h=1,stride_w=1):
    feature_dict={}
    (h,w)=gray_img.shape
    for i in range(0,h-lenh+1,stride_h):
        for j in range(0,w-lenw+1,stride_w):
            temp=gray_img[i:i+lenh,j:j+lenw]
            temp_int=integral_image(temp)
            bsum1=block_sum(temp_int,(0,0),int(lenw/2),lenh)
            bsum2=block_sum(temp_int,(int(lenw/2),0),int(lenw/2),lenh)
            res=bsum2-bsum1
            feature_dict['rect2_hort_'+str(lenh)+'x'+str(lenw)+'_'+'('+str(i)+','+str(j)+')']=res
    return feature_dict

def rect2_haar_vert(gray_img,lenh,lenw,stride_h=1,stride_w=1):
    feature_dict={}
    (h,w)=gray_img.shape
    for i in range(0,h-lenh+1,stride_h):
        for j in range(0,w-lenw+1,stride_w):
            temp=gray_img[i:i+lenh,j:j+lenw]
            temp_int=integral_image(temp)
            bsum1=block_sum(temp_int,(0,0),lenw,int(lenh/2))
            bsum2=block_sum(temp_int,(0,int(lenh/2)),lenw,int(lenh/2))
            res=bsum1-bsum2
            feature_dict['rect2_vert_'+str(lenh)+'x'+str(lenw)+'_'+'('+str(i)+','+str(j)+')']=res
    return feature_dict

def rect3_haar_hort(gray_img,lenh,lenw,stride_h=1,stride_w=1):
    feature_dict={}
    (h,w)=gray_img.shape
    for i in range(0,h-lenh+1,stride_h):
        for j in range(0,w-lenw+1,stride_w):
            temp=gray_img[i:i+lenh,j:j+lenw]
            temp_int=integral_image(temp)
            bsum1=block_sum(temp_int,(0,0),int(lenw/3),lenh)
            bsum2=block_sum(temp_int,(int(lenw/3),0),int(lenw/3),lenh)
            bsum3=block_sum(temp_int,(int((lenw*2)/3),0),int(lenw/3),lenh)
            res=bsum2-(bsum1+bsum3)
            feature_dict['rect3_hort_'+str(lenh)+'x'+str(lenw)+'_'+'('+str(i)+','+str(j)+')']=res
    return feature_dict

def rect4_haar_diag(gray_img,lenh,lenw,stride_h=1,stride_w=1):
    feature_dict={}
    (h,w)=gray_img.shape
    for i in range(0,h-lenh+1,stride_h):
        for j in range(0,w-lenw+1,stride_w):
            temp=gray_img[i:i+lenh,j:j+lenw]
            temp_int=integral_image(temp)
            bsum1=block_sum(temp_int,(0,0),int(lenw/2),int(lenh/2))
            bsum2=block_sum(temp_int,(int(lenw/2),0),int(lenw/2),int(lenh/2))
            bsum3=block_sum(temp_int,(0,int(lenh/2)),int(lenw/2),int(lenh/2))
            bsum4=block_sum(temp_int,(int(lenw/2),int(lenh/2)),int(lenw/2),int(lenh/2))
            res=bsum2+bsum3-(bsum1+bsum4)
            feature_dict['rect4_diag_'+str(lenh)+'x'+str(lenw)+'_'+'('+str(i)+','+str(j)+')']=res
    return feature_dict


if __name__ == "__main__":

    img=cv2.imread("C:\\Users\\tusha\\OneDrive\\Desktop\\facial_images\\0.png")
    print(img.shape)
    cv2.imshow("Img",img)
    cv2.waitKey(0)
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    (h,w)=gray.shape
    print(h,w)
    cv2.imshow("Gray",gray)
    cv2.waitKey(0)

    gray24=cv2.resize(gray, (24, 24),interpolation = cv2.INTER_AREA)
    (h24,w24)=gray24.shape
    print(h24,w24)
    cv2.imshow("Gray24",gray24)
    cv2.waitKey(0)
 

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
    
    
    print(len(features))

    
    
    
            
            

