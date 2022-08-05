#necessary imports
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import cv2
import numpy as np
import tflite_runtime.interpreter as tflite

imW = 620
imH = 480
#initiate the interpreter with the trained model
interpreter = tflite.Interpreter(model_path = 'model_combined_fp16.tflite')
interpreter.allocate_tensors()

#getting model detials
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

height = input_details[0]['shape'][1]
width = input_details[0]['shape'][2]

#check the type of input tensor
floating_model = (input_details[0]['dtype'] == np.float32)
input_mean = 127.5
input_std = 127.5
cap = cv2.VideoCapture('F:/boosto/Task4/bridge.mp4')
while True:
    vehicle_count = 0
    ret,frame = cap.read()
    frame = cv2.resize(frame,(imW,imH))
    frame_resized = cv2.resize(frame,(width,height))
    input_data = np.expand_dims(frame_resized,axis = 0)
    if floating_model:
        input_data = (np.float32(input_data)-input_mean)/input_std
    #performing detection on an image
    interpreter.set_tensor(input_details[0]['index'],input_data)
    interpreter.invoke()
    #retriveing detection results
    boxes = interpreter.get_tensor(output_details[0]['index'])[0] # Bounding box coordinates of detected objects
    classes = interpreter.get_tensor(output_details[1]['index'])[0] # Class index of detected objects
    scores = interpreter.get_tensor(output_details[2]['index'])[0] # Confidence of detected objects
    for i in range(len(scores)):
        if ((scores[i] > 0.3) and (scores[i] <= 1.0)):
            ymin = int(max(1,(boxes[i][0] * imH)))
            xmin = int(max(1,(boxes[i][1] * imW)))
            ymax = int(min(imH,(boxes[i][2] * imH)))
            xmax = int(min(imW,(boxes[i][3] * imW)))
            cv2.rectangle(frame, (xmin,ymin), (xmax,ymax), (10, 255, 0), 2)
            vehicle_count+=1
    cv2.putText(frame,'number of vehicles = ' + str(vehicle_count),(50,50),cv2.FONT_HERSHEY_SIMPLEX,1,255,4)
    cv2.imshow('frame',frame)
    if cv2.waitKey(1) == ord('q'):
        break
cv2.destroyAllWindows()
