import cv2
import numpy as np

cap = cv2.VideoCapture(-1)

classFile = 'coco.names'
classNames = []
with open(classFile, 'rt') as f:
    classNames = f.read().strip('\n').split('\n')

#print(classNames)

while True:
    success, img = cap.read()

    img = cv2.flip(img, 1)
    cv2.imshow('Image', img)
    cv2.waitKey(1)