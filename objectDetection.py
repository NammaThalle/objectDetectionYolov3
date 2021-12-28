import cv2
import numpy as np

cap = cv2.VideoCapture(-1)

width = 320
height = 320

CONFIDENCE_THRESHOLD = 0.5
NMS_THRESHOLD = 0.3         #lower the value more aggressive the working (i.e less number of boxes)

classFile = 'coco.names'
classNames = []
with open(classFile, 'rt') as f:
    classNames = f.read().strip('\n').split('\n')

# print(classNames)

modelConfiguration = 'misc/yolov3.cfg'
modelWeights = 'misc/yolov3.weights'

net = cv2.dnn.readNetFromDarknet(modelConfiguration, modelWeights)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

def findObjects(outputs, image):
    oHeight, oWeight, oCenter = image.shape
    boundingBox = []
    classIds = []
    confidenceLevels = []

    for output in outputs:
        for detection in output:
            scores = detection[5:]
            classId = np.argmax(scores)
            confidenceLevel = scores[classId]

            if confidenceLevel > CONFIDENCE_THRESHOLD:
                w, h = int(detection[2] * oWeight), int(detection[3] * oHeight)
                x, y = int((detection[0] * oWeight) - w / 2), int((detection[1] * oHeight) - h / 2)
                boundingBox.append([x, y, w, h])
                classIds.append(classId)
                confidenceLevels.append(float(confidenceLevel))

    # print(len(boundingBox))
    indices = cv2.dnn.NMSBoxes(boundingBox, confidenceLevels, CONFIDENCE_THRESHOLD, NMS_THRESHOLD)
    print(indices)

    for i in indices:
        # i = i[0]
        box = boundingBox[i]
        x, y, w, h = box[0], box[0], box[2], box[3]
        cv2.rectangle(image, (x,y), (x+w, y+h), (255, 0, 255), 2)
        cv2.putText(image, f'{classNames[classIds[i]].upper()} {int(confidenceLevels[i]*100)}%',
                    (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6,(255, 0, 255), 2)

while True:
    success, img = cap.read()

    # img = cv2.flip(img, 1)

    blob = cv2.dnn.blobFromImage(img, 1/255, (width, height), [0,0,0], 1, crop = False)
    net.setInput(blob)

    totalLayers = net.getLayerNames()
    # print(type(net.getUnconnectedOutLayers()))
   
    outputLayers = [totalLayers[i-1] for i in net.getUnconnectedOutLayers()] 
    # print(outputLayers)

    outputs = net.forward(outputLayers)
    # print(len(outputs))
    # print(outputs[0].shape)
    # print(outputs[1].shape)
    # print(outputs[2].shape)
    # print(outputs[0][0])

    findObjects(outputs, img)

    cv2.imshow('Image', img)
    cv2.waitKey(1)