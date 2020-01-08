import cv2
import numpy as np
import time
import speech_recognition as sr
import speech_to_text
import threading
import pyttsx3
engine = pyttsx3.init()
engine.setProperty('rate', 120)
myName = 'hello'
currentClassDetecting = 'background'
currentIndicesDectecting = 20
coun = 2

def run_video_detection(scoreThreshold):
    #Load YOLO v3
    classes_val = {}
    weights = "/home/anuj/PycharmProjects/726/custom/darknet-yolov3_1500.weights"
    cfg ="/home/anuj/PycharmProjects/726/custom/darknet-yolov3.cfg"
    nam = "/home/anuj/PycharmProjects/726/custom/classes.names"
    net = cv2.dnn.readNet(weights, cfg) # Original yolov3
    global classes
    global class_names
    classes = {}
    class_names = {}
    with open(nam,"r") as f:
        classes_list = [line.strip() for line in f.readlines()]
        class_names = {i: classes_list[i] for i in range(0, len(classes_list))}

    layer_names = net.getLayerNames()
    outputlayers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    cap = cv2.VideoCapture(0)
    font = cv2.FONT_HERSHEY_PLAIN
    starting_time = time.time()
    frame_id = 0

    while True:
        _, frame = cap.read()  #
        frame_id += 1

        height, width, channels = frame.shape
        blob = cv2.dnn.blobFromImage(frame, 0.00392, (256, 256), (0, 0, 0), True, crop=False)  # reduce 416 to 320

        net.setInput(blob)
        outs = net.forward(outputlayers)
        classes[currentIndicesDectecting] = currentClassDetecting
        list_of_classes = [currentIndicesDectecting]
        
        for j in list_of_classes:
            count = 0
            if j in classes:
                lab = classes[j]
            else:
                lab = 'background'
        
            class_ids = []
            confidences = []
            boxes = []
            for out in outs:
                for detection in out:
                    scores = detection[5:]
                    class_id = np.argmax(scores)
                    confidence = scores[class_id]
                    if class_id == j and confidence > 0.3:
                        center_x = int(detection[0] * width)
                        center_y = int(detection[1] * height)
                        w = int(detection[2] * width)
                        h = int(detection[3] * height)
                        x = int(center_x - w / 2)
                        y = int(center_y - h / 2)
                       

                        boxes.append([x, y, w, h])  # put all rectangle areas
                        confidences.append(
                            float(confidence))  
                        class_ids = [class_id]  # name of the object that was detected
            
            for i in range(len(boxes)):
                
                x, y, w, h = boxes[i]
                label = str(classes[class_ids[0]])
                confidence = confidences[i]
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, label + " " + str(round(confidence, 2)), (x, y + 30), font, 1, (255, 255, 255), 2)
                global coun
                if coun == 1 and lab == currentClassDetecting:
                    engine.say(str(currentClassDetecting) + "FOUND")
                    engine.runAndWait()
                    engine.stop()
                    coun = 2
            
        cv2.imshow("Live Camera for Custom Object Detection", frame)
        key = cv2.waitKey(1) 

        if key == 27: 
            break;

    cap.release()
    cv2.destroyAllWindows()

def get_key(val):
    for key, value in class_names.items():
        if val == value:
            return int(key)

    return "key doesn't exist"


def run_voice_command(showVideoStream):
    global currentClassDetecting
    global currentIndicesDectecting
    global coun
    currentClassDetecting = 'background'
    currentIndicesDectecting = 20

    rc = sr.Recognizer()
    while showVideoStream:
        rc.energy_threshold = 60
        mic = sr.Microphone()

        with mic as source:
            rc.adjust_for_ambient_noise(source, duration=0.5)
            print("Adjustment Completed")
            result = speech_to_text.convert_to_text()
            print("result: ",result)
            if result is not None:
                if result == myName:
                    print('Say command')
                    time.sleep(2)
                    obj = speech_to_text.convert_to_text()
                    print(obj)

                    if obj in class_names.values():
                        print("Object in Class")
                        currentClassDetecting = obj
                        indices = get_key(obj)
                        coun = 1
                        currentIndicesDectecting = indices
                        print('Now detecting: ' + obj)
                        
                    else:
                        print('The object ' + str(obj) + ' is invalid')
                        currentIndicesDectecting = 20
                        
                else:
                    print("to start, say Hello")
                    pass

pass

def detect(voice_cmd,score_threshold,showVideoStream,currentClass,currentIndices):

    global currentClassDetecting
    global currentIndicesDectecting
    currentClassDetecting = currentClass
    currentIndicesDectecting = currentIndices

    if voice_cmd == True:
        voiceCommandThread = threading.Thread(target=run_voice_command, args=[showVideoStream])
        voiceCommandThread.start()

    videoStreamThread = threading.Thread(target=run_video_detection,
                                         args=[score_threshold])
    videoStreamThread.start()

