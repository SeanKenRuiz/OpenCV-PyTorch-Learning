import cv2 as cv
import numpy as np


img = cv.imread('testTubeAngled.PNG')

gray = cv.imread('testTubeAngled.png', cv.IMREAD_GRAYSCALE)

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

laplacian = cv.Laplacian(d3,cv.CV_64F)
sobelx = cv.Sobel(d3,cv.CV_64F,1,0,ksize=5)
sobely = cv.Sobel(d3,cv.CV_64F,0,1,ksize=5)

abs_sobelx64f = np.absolute(sobelx)
abs_sobely64f = np.absolute(sobely)
sobelx_8u = np.uint8(abs_sobelx64f)
sobely_8u = np.uint8(abs_sobely64f)

cv.imshow('frame', gray)
cv.imshow('laplacian', laplacian)
cv.imshow('sobelx', sobelx)
cv.imshow('sobely', sobely)

cv.waitKey(0)
