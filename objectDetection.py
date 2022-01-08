import cv2
import numpy as np
import time

# to stream in from a webcam
# cap = cv2.VideoCapture(-1)

# default constant 
# width and height for converting each frame into the blob
width = 320
height = 320

# confidence and nms threshold value to find out max confidence among a given set
CONFIDENCE_THRESHOLD = 0.5
NMS_THRESHOLD = 0.3         #lower the value more aggressive the working (i.e less number of boxes)

classFile = 'coco.names'

# used to record the time when we processed last frame
prev_frame_time = 0
 
# used to record the time at which we processed current frame
new_frame_time = 0

# list to store classifier names
classNames = []

# configuration and weight file paths for the respective yolov3 network
# yolov3-320 model
modelConfiguration = 'misc/yolov3.cfg'
modelWeights = 'misc/yolov3.weights'

# yolov3-tiny model
# modelConfiguration = 'misc/yolov3-tiny.cfg'
# modelWeights = 'misc/yolov3-tiny.weights'

# to stream in from a video
cap = cv2.VideoCapture("52M27S_1640863347.mp4")
 
# open the file and read the class names 
with open(classFile, 'rt') as f:
    classNames = f.read().strip('\n').split('\n')

# create neural network using the model configurations and
# set the different targets
net = cv2.dnn.readNetFromDarknet(modelConfiguration, modelWeights)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

# function to find the objects in an image
def findObjects(outputs, image):
    oHeight, oWidth, oCenter = image.shape
    boundingBox = []
    classIds = []
    confidenceLevels = []

    for output in outputs:
        for detection in output:
            scores = detection[5:]
            classId = np.argmax(scores)
            confidenceLevel = scores[classId]

            if confidenceLevel > CONFIDENCE_THRESHOLD:
                w, h = int(detection[2] * oWidth), int(detection[3] * oHeight)
                x, y = int((detection[0] * oWidth) - w / 2), int((detection[1] * oHeight) - h / 2)
                boundingBox.append([x, y, w, h])
                classIds.append(classId)
                confidenceLevels.append(float(confidenceLevel))

    indices = cv2.dnn.NMSBoxes(boundingBox, confidenceLevels, CONFIDENCE_THRESHOLD, NMS_THRESHOLD)
    print(indices)

    for i in indices:
        # i = i[0]
        box = boundingBox[i]
        x, y, w, h = box[0], box[0], box[2], box[3]
        cv2.rectangle(image, (x,y), (x+w, y-h), (255, 0, 255), 2)
        cv2.putText(image, f'{classNames[classIds[i]].upper()} {int(confidenceLevels[i]*100)}%',
                    (x, y-h-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6,(255, 0, 255), 2)

# loop till user inputs a q
while True:
    success, img = cap.read()

    if img is None:
        print("wrong path")

    # img = cv2.flip(img, 1)
    new_frame_time = time.time()

    # Calculating the fps
    # fps will be number of frame processed in given time frame
    # since their will be most of time error of 0.001 second
    # we will be subtracting it to get more accurate result
    fps = 1/(new_frame_time-prev_frame_time)
    prev_frame_time = new_frame_time

    # converting the fps into integer
    fps = int(fps)
 
    # converting the fps to string so that we can display it on frame
    # by using putText function
    fps = str(fps)
 
    # putting the FPS count on the frame
    cv2.putText(img, fps, (7, 120), cv2.FONT_HERSHEY_SIMPLEX, 3, (100, 255, 0), 3, cv2.LINE_AA)

    blob = cv2.dnn.blobFromImage(img, 1/255, (width, height), [0,0,0], 1, crop = False)
    net.setInput(blob)

    totalLayers = net.getLayerNames()

    outputLayers = [totalLayers[i-1] for i in net.getUnconnectedOutLayers()] 
    outputs = net.forward(outputLayers)

    findObjects(outputs, img)

    cv2.imshow('Image', img)

    # if user inputs q, break
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break