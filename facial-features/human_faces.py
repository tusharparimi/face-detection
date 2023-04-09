import cv2
import numpy as np

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

#smoothing the image
blurred_roi=cv2.GaussianBlur(roi,(3,3),0)
cv2.imshow("Blurred",blurred_roi)
cv2.waitKey(0)

#applying Canny edge detection
edged_roi=cv2.Canny(gray_roi,30,150)
cv2.imshow("Edged",edged_roi)
cv2.waitKey(0)

#applying Canny edge detection
edged_roi_s=cv2.Canny(blurred_roi,30,150)
cv2.imshow("Edged_smooth",edged_roi_s)
cv2.waitKey(0)

#storing all possible roi of the complete image in a list
(rh,rw,rd)=roi.shape
roi_list=[]
for i in range(0,h-rh):
    for j in range(0,w-rw):
        temp=image[i:rh,j:rw]
        roi_list.append(temp)
print("count of roi: ",len(roi_list))

def integral_image(img):
    h=img.shape[0]
    w=img.shape[1]
    #int_img=np.zeros((h,w))
    int_img = [ [ 0 for y in range(w) ] for x in range(h) ]
    print(img.shape)
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

roi_int=integral_image(gray_roi)
print(len(roi_int))
print(len(roi_int[0]))
print(roi.shape)

def block_sum(int_img,p,lenx,leny):
    h=len(int_img)
    w=len(int_img[0])
    print(h,w)
    print(p[0]+lenx,p[1]+leny)
    print(int_img[0][0])
    print(int_img[p[1]+leny-1][p[0]+lenx-1])
    if(p[0]+lenx<=w and p[1]+leny<=h):
        bsum=int_img[p[1]+leny-1][p[0]+lenx-1]+max(0,int_img[p[1]-1][p[0]-1])-max(0,int_img[p[1]-1][p[0]+lenx-1])-max(0,int_img[p[1]+leny-1][p[0]-1])
        return bsum
    else:
        print("point and length combo out of bound")
print(roi_int[3][3])
print(block_sum(roi_int,[0,0],3,3))
    


    
    








