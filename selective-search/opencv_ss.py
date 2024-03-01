import cv2
import numpy as np
import time
import random

img=cv2.imread("selective-search\\test.jpg")
print(img.shape)

cv2.imshow("img", img)
cv2.waitKey(0)

ss=cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
ss.setBaseImage(img)

ss.switchToSelectiveSearchFast()
#ss.switchToSelectiveSearchQuality()

start=time.time()
rects=ss.process()
end=time.time()

print(f"time for selective search: {round(end-start, 4)} sec")
print(f"Total regions proposals: {len(rects)}")

print(rects[0])


for i in range(0, len(rects)):
    res=img.copy()
    for (x, y, w, h) in rects[i:i+100]:
        color = [random.randint(0, 255) for j in range(0, 3)]
        cv2.rectangle(res, (x, y), (x + w, y + h), color, 2)
    
    cv2.imshow("res", res)
    k=cv2.waitKey(1)
    if k==ord("q"):
        break
