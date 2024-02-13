import cv2
import numpy as np

def integral_image(img):
    h=img.shape[0]
    w=img.shape[1]
    #int_img=np.zeros((h,w))
    int_img = [ [ 0 for y in range(w) ] for x in range(h) ]
    #print(img.shape)
    #print(int_img.shape)
    #print(integral_image.shape)
    for y in range(0, h):
            sum = 0
            for x in range(0, w):
                sum += img[y][x]
                int_img[y][x] = sum
                if y > 0:
                    int_img[y][x] += int_img[y-1][x]
    return int_img



def block_sum(int_img,p,lenx,leny):
    h=len(int_img)
    w=len(int_img[0])
    #print(h,w)
    #rint(p[0]+lenx-1,p[1]+leny-1)
    #print(int_img[0][0])
    #print(int_img[p[1]+leny-1][p[0]+lenx-1])
    if(p[0]+lenx<=w and p[1]+leny<=h):
        bsum=int_img[p[1]+leny-1][p[0]+lenx-1]
        if p[1]-1>=0 and p[0]-1>=0:
             bsum+=int_img[p[1]-1][p[0]-1]-int_img[p[1]-1][p[0]+lenx-1]-int_img[p[1]+leny-1][p[0]-1]
        return bsum
    else:
        print("point and length combo out of bound")


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

    #converting to grayscale
    gray_roi=cv2.cvtColor(roi,cv2.COLOR_BGR2GRAY)
    cv2.imshow("Gray",gray_roi)
    cv2.waitKey(0)
    print("Gray ROI shape: ",gray_roi.shape)

    roi_int=integral_image(gray_roi)
    print(len(roi_int))
    print(len(roi_int[0]))
    print(roi.shape)

    print(roi_int[0][0:3])
    print(roi_int[1][0:3])
    print(roi_int[2][0:3])
    print(roi_int[3][0:3])
    print(roi_int[2][2])
    print(roi_int[0][2])
    print(roi_int[2][0])
    print(roi_int[0][0])
    print(block_sum(roi_int,[1,1],3,2))

