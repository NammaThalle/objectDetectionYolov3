import cv2
import numpy as np

import threading
import queue

from imutils.video import FPS

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

# variable to store the fps details of the video
fps = None
def APP_LOG (logString):
    print("[ LOG ] : "+ logString)

def APP_ERROR(errorString):
    print("[ ERROR ] "+ errorString)

def imagePreprocess(cap, frameQueue, blobQueue, width, height):
    global fps
    while cap.isOpened():
        success, frame = cap.read()

        if frame is None or success is False:
            APP_ERROR("IMAGE READ FAILED")
            break

        fps = FPS().start()
        frame = cv2.resize(frame, (int(frame.shape[1] * 0.5),int(frame.shape[0] * 0.5)))

        blob = cv2.dnn.blobFromImage(frame, 1/255, (width, height), [0,0,0], 1, crop = False)

        frameQueue.put(frame)
        blobQueue.put(blob)

# function to find the objects in an image
def findObjects(outputs, frame):
    oHeight, oWidth, oCenter = frame.shape
    boundingBox = []
    classIds = []
    confidenceLevels = []

    global classNames
    global fps

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
    APP_LOG("OBJECTS DETECTED : " + str(len(indices)))

    for i in indices:
        # i = i[0]
        box = boundingBox[i]
        x, y, w, h = box[0], box[0], box[2], box[3]
        cv2.rectangle(frame, (x,y), (x+w, y-h), (255, 0, 255), 2)
        cv2.putText(frame, f'{classNames[classIds[i]].upper()} {int(confidenceLevels[i]*100)}%',
                    (x, y-h-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6,(255, 0, 255), 2)

def imageProcessing(cap, net, frameQueue, blobQueue):

    global fps

    # loop till user inputs a q
    while True:
        blob = blobQueue.get()

        net.setInput(blob)

        totalLayers = net.getLayerNames()

        outputLayers = [totalLayers[i-1] for i in net.getUnconnectedOutLayers()] 
        outputs = net.forward(outputLayers)

        frame = frameQueue.get()
        findObjects(outputs, frame)

        cv2.imshow('Image', frame)

        fps.update()

        # if user inputs q, break
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break        

    fps.stop()
    APP_LOG(" APPROX FPS : {:.2f}".format(fps.fps()))

def main():

    global classNames
    global fps

    # to stream in from a video
    APP_LOG("STARTING VIDEO CAPTURE")
    cap = cv2.VideoCapture("52M27S_1640863347.mp4")
    cap.open("52M27S_1640863347.mp4")
    APP_LOG("STARTED VIDEO CAPTURE")

    frameQueue = queue.Queue(maxsize=4)
    blobQueue = queue.Queue(maxsize=4)

    # open the file and read the class names 
    with open(classFile, 'rt') as f:
        classNames = f.read().strip('\n').split('\n')

    # create neural network using the model configurations and
    # set the different targets
    APP_LOG("LOADING NN MODEL")
    net = cv2.dnn.readNetFromDarknet(modelConfiguration, modelWeights)
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

    imagePreprocessThread = threading.Thread(target=imagePreprocess, 
                    args=(cap, frameQueue, blobQueue, width, height))
    imageProcessThread = threading.Thread(target = imageProcessing,
                    args=(cap, net, frameQueue, blobQueue))

    imagePreprocessThread.start()                    
    # imageProcessThread.start()
    imageProcessing(cap, net, frameQueue, blobQueue)
    imagePreprocessThread.join()
    # imageProcessThread.join()

    APP_LOG("STOPPING VIDEO CAPTURE")
    cap.release()
    APP_LOG("STOPPED VIDEO CAPTURE")
    cv2.destroyAllWindows()

if __name__=="__main__":
    main()