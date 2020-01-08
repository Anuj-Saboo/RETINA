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

                        #take indices from dictionary
                        currentIndicesDectecting = indices
                        print('Now detecting: ' + obj)

                        coun = 1

                    else:
                        print('The object ' + str(obj) + ' is invalid')
                        currentIndicesDectecting = 20
                        currentClassDetecting = "background"

                else:
                    print("to start, say Hello")
                    pass

pass


def run_video_detection(scoreThreshold):

    classes = {}
    inputs = tf.placeholder(tf.float32, [None, 256, 256, 3])
    model = nets.YOLOv3COCO(inputs, nets.Darknet19)
    with tf.Session() as sess:
        sess.run(model.pretrained())
        cap = cv2.VideoCapture(0)

        while (cap.isOpened()):

            classes[currentIndicesDectecting] = currentClassDetecting
            list_of_classes = [currentIndicesDectecting]
            ret, frame = cap.read()
            img = cv2.resize(frame, (256, 256))
            imge = np.array(img).reshape(-1, 256, 256, 3)
            start_time = time.time()
            preds = sess.run(model.preds, {inputs: model.preprocess(imge)})

            boxes = model.get_boxes(preds, imge.shape[1:3])
            cv2.namedWindow('Live Camera', cv2.WINDOW_NORMAL)

            cv2.resizeWindow('Live Camera', 500, 500)
            boxes1 = np.array(boxes)

            for j in list_of_classes:
                count = 0
                
                if j in classes:
                    lab = classes[j]  
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
        
