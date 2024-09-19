import numpy as np
import cv2 as cv

img1 = cv.imread('messi.jpg')

e1 = cv.getTickCount()

# Code running
for i in range(5,10,2):
    img1 = cv.medianBlur(img1,i)
# Code end

cv.imshow('image', img1)

cv.waitKey(0)

e2 = cv.getTickCount()
time = (e2-e1) / cv.getTickFrequency()

print(time)