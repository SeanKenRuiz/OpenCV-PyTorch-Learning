import cv2 as cv
import numpy as np


img = cv.imread('testTubeAngled.PNG')

gray = cv.imread('testTubeAngled.png', cv.IMREAD_GRAYSCALE)

blur = cv.GaussianBlur(gray, (5,5), 0)

ret, th1 = cv.threshold(blur, 127, 255, cv.THRESH_BINARY)
th2 = cv.adaptiveThreshold(blur, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 11, 2)
th3 = cv.adaptiveThreshold(blur, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 11, 2)

cv.imshow('frame', gray)
cv.imshow('th1', th1)
cv.imshow('th2', th2)
cv.imshow('th3', th3)

cv.waitKey(0)
