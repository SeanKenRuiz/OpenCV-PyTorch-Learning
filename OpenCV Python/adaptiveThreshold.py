import cv2 as cv
import numpy as np

cap = cv.VideoCapture(0)

while True:
    # Take each frame
    _, frame = cap.read()

    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    blur = cv.medianBlur(gray, 5)

    ret, th1 = cv.threshold(blur, 127, 255, cv.THRESH_BINARY)
    th2 = cv.adaptiveThreshold(blur, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 11, 2)
    th3 = cv.adaptiveThreshold(blur, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 11, 2)

    cv.imshow('frame', gray)
    cv.imshow('th1', th1)
    cv.imshow('th2', th2)
    cv.imshow('th3', th3)

    if cv.waitKey(1) == ord('q'):
        break
