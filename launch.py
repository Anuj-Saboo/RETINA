import threading
import numpy as np
import speech_recognition as sr
import speech_to_text
import tensorflow as tf
import tensornets as nets
import cv2
import time
import pyttsx3
engine = pyttsx3.init()
engine.setProperty('rate', 120)


cvNet = None
myName = 'hello'
showVideoStream = False

audio_yes = 'audio/yes.wav'
audio_okay = 'audio/okay.wav'
audio_invalid = 'audio/invalid.wav'


classNames={0:'person',2:'car',15:'cat',16:'dog',24:'backpack',25:'umbrella',26:'handbag',39:'bottle',46:'banana',
            47:'apple',49:'orange',53:'pizza',55:'donut',63:'laptop',64:'mouse',66:'keyboard',
            67:'cell phone',68:'microwave',79:'toothbrush'}


# def create_capture(source=0):
#     source = str(source).strip()
#     chunks = source.split(':')
#     # handle drive letter ('c:', ...)
#     if len(chunks) > 1 and len(chunks[0]) == 1 and chunks[0].isalpha():
#         chunks[1] = chunks[0] + ':' + chunks[1]
#         del chunks[0]
#
#     source = chunks[0]
#     try:
#         source = int(source)
#     except ValueError:
#         pass
#     params = dict(s.split('=') for s in chunks[1:])
#
#     cap = cv.VideoCapture(source)
#     if 'size' in params:
#         w, h = map(int, params['size'].split('x'))
#         cap.set(cv.CAP_PROP_FRAME_WIDTH, w)
#         cap.set(cv.CAP_PROP_FRAME_HEIGHT, h)
#     if cap is None or not cap.isOpened():
#         print('Warning: unable to open video source: ', source)
#     return cap

"""
def label_class(img, detection, score, className, boxColor=None):
    Height = img.shape[0]
    Width = img.shape[1]

    if boxColor == None:
        boxColor = (23, 230, 210)
    center_x = int(detection[0] * Width)
    center_y = int(detection[1] * Height)
    w = int(detection[2] * Width)
    h = int(detection[3] * Height)
    x = center_x - w / 2
    y = center_y - h / 2
    cv.rectangle(img, (round(x), round(y)), (round(x+w), round(y+h)), boxColor, thickness=4)

    label = className + ": " + str(int(round(score * 100))) + '%'
    labelSize, baseLine = cv.getTextSize(label, cv.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    y = max(y, labelSize[1])
    cv.rectangle(img, (x, y - labelSize[1]), (x + labelSize[0], y + baseLine),
                 (255, 255, 255), cv.FILLED)
    cv.putText(img, label, (x, y), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))
    pass


def detect_object(img, detections, score_threshold, classNames, className):
    #print(className)
    #print(classNames)
    for out in detections:
        for detection in out:
            #print("detections")
            #print(detection)
            score = detection[5:]
            #print("score")
            #print(score)
            class_id = np.argmax(score)
            #print("class id ", class_id)
            confidence = score[class_id]
            if className in classNames.values() and className == classNames[class_id] and confidence > score_threshold:
                print(confidence)
                label_class(img, detection, confidence, classNames[class_id])
        pass

def play_audio(audioFile):
    chunk = 1024
    wf = wave.open(audioFile, 'rb')
    pa = pyaudio.PyAudio()
    stream = pa.open(format=pa.get_format_from_width(wf.getsampwidth()),
                     channels=wf.getnchannels(),
                     rate=wf.getframerate(),
                     output=True)

    data = wf.readframes(chunk)
    while len(data) > 0:
        stream.write(data)
        data = wf.readframes(chunk)

    stream.stop_stream()
    stream.close()
    pa.terminate()
"""

def get_key(val):
    for key, value in classNames.items():
        if val == value:
            return int(key)

    return "key doesn't exist"


def run_voice_command():
    rc = sr.Recognizer()
    while showVideoStream:
        rc.energy_threshold = 60
        mic = sr.Microphone()

        with mic as source:
            print("I am Adjusting for Noise")
            rc.adjust_for_ambient_noise(source,duration=0.5)
            print("Adjustment Completed")


            result = speech_to_text.convert_to_text()
            if result is not None:
                if result == myName:
                    #print('Say command')
                    time.sleep(2)
                    obj = speech_to_text.convert_to_text()
                    print(obj)

                    if obj in classNames.values():
                        print("Object in Class")
                        global currentClassDetecting
                        global currentIndicesDectecting
                        global coun
                        currentClassDetecting = obj
                        indices = get_key(obj)

                        #indices = classNames[obj]
                        #take indices from dictionary
                        currentIndicesDectecting = indices
                        print('Now detecting: ' + obj)

                        coun = 1
                        #play_audio(audio_okay)
                    else:
                        print('The object ' + str(obj) + ' is invalid')
                        currentIndicesDectecting = 20
                        currentClassDetecting = "background"
                        #play_audio(audio_invalid)


                else:
                    print("to start, say Hello")
                    pass # ignore unrecognizable audio

#print('exiting run_voice_command...')
pass


def run_video_detection(scoreThreshold):

    classes = {}
    inputs = tf.placeholder(tf.float32, [None, 256, 256, 3])
    model = nets.YOLOv3COCO(inputs, nets.Darknet19)
    with tf.Session() as sess:
        sess.run(model.pretrained())
        cap = cv2.VideoCapture(0)

        while (cap.isOpened()):

            #print(classes)
            classes[currentIndicesDectecting] = currentClassDetecting
            list_of_classes = [currentIndicesDectecting]
            #print(list_of_classes)
            ret, frame = cap.read()
            img = cv2.resize(frame, (256, 256))
            imge = np.array(img).reshape(-1, 256, 256, 3)
            start_time = time.time()
            preds = sess.run(model.preds, {inputs: model.preprocess(imge)})

            #print("--- %s seconds ---" % (time.time() - start_time))
            boxes = model.get_boxes(preds, imge.shape[1:3])
            cv2.namedWindow('Live Camera', cv2.WINDOW_NORMAL)

            cv2.resizeWindow('Live Camera', 500, 500)
            # print("--- %s seconds ---" % (time.time() - start_time))
            boxes1 = np.array(boxes)

            for j in list_of_classes:
                #print(j)
                #print(type(j))
                count = 0
                #lab = 'elephant'
                if j in classes:
                    #print("IN")

                    lab = classes[j]
                    #print(boxes1[j])
                    #print(lab)
                else:
                    lab = 'background'
                if len(boxes1) != 0:

                    for i in range(len(boxes1[j])):
                        box = boxes1[j][i]

                        if boxes1[j][i][4] >= .40:
                            count += 1
                            obj = lab


                            cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 1)
                            cv2.putText(img, lab, (box[0], box[1]), cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 0, 255),
                                        lineType=cv2.LINE_AA)
                            global coun
                            if coun == 1  and lab == currentClassDetecting:
                                engine.say(str(currentClassDetecting) + "FOUND")
                                engine.runAndWait()
                                engine.stop()
                                coun = 2
                #print(lab, ": ", count)

            cv2.imshow("Live Camera", img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break


    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    import sys
    import argparse
    import custom_yolo
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",
                        help="1 for COCO dataset and 2 for custom objects",
                        type=int, default=1)
    parser.add_argument("--voice_cmd", help="Enable voice commands", default=True)
    parser.add_argument("--score_threshold",
                        help="Only show detections with a probability of correctness above the specified threshold",
                        type=float, default=0.3)
    parser.add_argument("--currentclass",help="Value in case don't want to use voice",type=str,default="background")
    parser.add_argument("--currentindice", help="Value in case don't want to use voice", type=int, default=20)
    args = parser.parse_args()

    currentClassDetecting = args.currentclass
    currentIndicesDectecting = args.currentindice
    coun = 2

    showVideoStream = True
    if args.model ==1:


        videoStreamThread = threading.Thread(target=run_video_detection,
                                             args=[args.score_threshold])
        videoStreamThread.start()
        time.sleep(2)
        if args.voice_cmd == True:
            voiceCommandThread = threading.Thread(target=run_voice_command)
            voiceCommandThread.start()

    elif args.model == 2:
        custom_yolo.detect(args.voice_cmd,args.score_threshold,showVideoStream,currentClassDetecting,currentIndicesDectecting)
        #out = custom_yolo.custom_video_detection
        #voic = custom_yolo.custom_audio_detection
