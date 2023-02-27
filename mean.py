import cv2
import numpy as np

cap = cv2.VideoCapture("D:/python/DATA/10_20.mp4")


success, sum = cap.read()
sum = sum.astype(np.float32)
sum = sum/300
COUNT = 0
while success:
    if COUNT == 299: break
    success, img1 = cap.read()
    img1 = img1.astype(np.float32)
    COUNT += 1
    sum += img1/300
    print(COUNT)
    # print(sum)
sum = sum.astype(np.uint8)
cv2.imwrite("./back.jpg".format(np), sum)
cv2.imshow('background', sum)
cv2.waitKey()
