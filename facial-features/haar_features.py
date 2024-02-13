import cv2
import numpy as np
from integral_image import integral_image, block_sum

def edge_haar_lr(img,gray_img,lenh,lenw,threshold):
    out=img.copy()
    (h,w)=gray_img.shape
    for i in range(0,h-lenh,lenh):
        for j in range(0,w-lenw,lenw):
            temp=gray_img[i:i+lenh,j:j+lenw]
            temp_int=integral_image(temp)
            bsum1=block_sum(temp_int,(0,0),int(lenw/2),lenh)
            bsum2=block_sum(temp_int,(int(lenw/2),0),int(lenw/2),lenh)
            #print("block sums:")
            #print(bsum1/((lenh/2)*(lenw/2)))
            #print(bsum2/((lenh/2)*(lenw/2)))
            if abs((bsum1/((lenh/2)*(lenw/2)))-(bsum2/((lenh/2)*(lenw/2))))>=threshold:
                cv2.rectangle(out,(j,i),(j+int(lenw/2)-1,i+lenh-1),(0,250,0),1)
                cv2.rectangle(out,(j+int(lenw/2),i),(j+lenw-1,i+lenh-1),(0,0,250),1)
    return out

def edge_haar_ud(img,gray_img,lenh,lenw,threshold):
    out=img.copy()
    (h,w)=gray_img.shape
    for i in range(0,h-lenh,lenh):
        for j in range(0,w-lenw,lenw):
            temp=gray_img[i:i+lenh,j:j+lenw]
            temp_int=integral_image(temp)
            bsum1=block_sum(temp_int,(0,0),lenw,int(lenh/2))
            bsum2=block_sum(temp_int,(0,int(lenh/2)),lenw,int(lenh/2))
            #print("block sums:")
            #print(bsum1/((lenh/2)*(lenw/2)))
            #print(bsum2/((lenh/2)*(lenw/2)))
            if abs((bsum1/((lenh/2)*(lenw/2)))-(bsum2/((lenh/2)*(lenw/2))))>=threshold:
                cv2.rectangle(out,(j,i),(j+lenw-1,i+int(lenh/2)-1),(250,0,0),1)
                cv2.rectangle(out,(j,i+int(lenh/2)),(j+lenw-1,i+lenh-1),(0,0,250),1)
    return out


if __name__ == "__main__":

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

    roi1=image[580:720,120:260]
    print("ROI1 shape: ",roi1.shape)
    cv2.imshow("ROI1",roi1)
    cv2.waitKey(0)

    #converting to grayscale
    gray_roi=cv2.cvtColor(roi,cv2.COLOR_BGR2GRAY)
    cv2.imshow("Gray",gray_roi)
    cv2.waitKey(0)
    print("Gray ROI shape: ",gray_roi.shape)

    gray_roi1=cv2.cvtColor(roi1,cv2.COLOR_BGR2GRAY)
    cv2.imshow("Gray1",gray_roi1)
    cv2.waitKey(0)
    print("Gray ROI1 shape: ",gray_roi1.shape)

    v1=edge_haar_lr(roi,gray_roi,20,20,255)
    cv2.imshow("v1",v1)
    cv2.waitKey(0)
    h1=edge_haar_ud(roi,gray_roi,20,40,255)
    cv2.imshow("h1",h1)
    cv2.waitKey(0)
    v2=edge_haar_lr(roi1,gray_roi1,20,20,150)
    cv2.imshow("v2",v2)
    cv2.waitKey(0)
    h2=edge_haar_ud(roi1,gray_roi1,20,40,150)
    cv2.imshow("h2",h2)
    cv2.waitKey(0)

    for i in range(1,140):
        for j in range(2,140):
            v1=edge_haar_lr(roi,gray_roi,i,j,255)
            cv2.imshow("v1",v1)
            cv2.waitKey(0)

    for i in range(1,140):
        for j in range(2,140):
            v2=edge_haar_lr(roi1,gray_roi1,i,j,150)
            cv2.imshow("v2",v2)
            cv2.waitKey(0)



    for i in range(2,140):
        for j in range(1,140):
            h1=edge_haar_ud(roi,gray_roi,i,j,255)
            cv2.imshow("h1",h1)
            cv2.waitKey(0)

    for i in range(2,140):
        for j in range(1,140):
            h2=edge_haar_ud(roi1,gray_roi1,i,j,150)
            cv2.imshow("h2",h2)
            cv2.waitKey(0)

                

        
