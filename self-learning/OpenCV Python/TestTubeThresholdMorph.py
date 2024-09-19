import cv2 as cv
import numpy as np


img = cv.imread('testTube.PNG')

gray = cv.imread('testTube.png', cv.IMREAD_GRAYSCALE)

blur = cv.GaussianBlur(gray, (5,5), 0)

ret, th1 = cv.threshold(blur, 127, 255, cv.THRESH_BINARY)
th2 = cv.adaptiveThreshold(blur, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 11, 2)
th3 = cv.adaptiveThreshold(blur, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 11, 2)

kernel_0 = np.zeros((5,5),np.uint8)
kernel_1 = np.ones((3,3),np.uint8)

d1 = cv.dilate(th1, kernel_1, iterations = 1)
d2 = cv.dilate(th2, kernel_1, iterations = 1)
d3 = cv.dilate(th3, kernel_1, iterations = 1)

d1 = cv.dilate(d1, kernel_0, iterations = 5)
d2 = cv.dilate(d2, kernel_0, iterations = 5)
d3 = cv.dilate(d3, kernel_0, iterations = 5)

spl_image1 = np.hstack((th1, d1))
spl_image2 = np.hstack((th2, d2))
spl_image3 = np.hstack((th3, d3))

cv.imshow('frame', gray)
cv.imshow('th1', spl_image1)
cv.imshow('th2', spl_image2)
cv.imshow('th3', spl_image3)

cv.waitKey(0)
