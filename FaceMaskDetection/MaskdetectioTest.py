# -*- coding: utf-8 -*-
"""
Created on Wed Dec  9 13:20:46 2020

@author: Borun Das
"""

import tensorflow
from tensorflow import keras
from keras.models import load_model
from tensorflow.keras.preprocessing import image
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


print(tensorflow.__version__)
print(keras.__version__)



model = load_model('model/maske_detection_model_phase2.h5')
face_cascade=cv2.CascadeClassifier('haarscadefile/haarcascade_frontalface_default.xml')



def detectface(img):
    face_img = img.copy()
    #face_img = cv2.cvtColor(face_img,cv2.COLOR_BGR2GRAY)
    face_rects= face_cascade.detectMultiScale(face_img, scaleFactor=1.1,
        minNeighbors=5,minSize=(30, 30),flags=cv2.CASCADE_SCALE_IMAGE)
    for (x,y,w,h) in face_rects:
        roi=face_img[y:y+h,x:x+w]
        preidiction,prediction_class=predict_single_img(roi)
        """
        cv2.imwrite('demo_output/roi.jpg', roi) 
        filename='demo_output/roi.jpg'
        preidiction,prediction_class=predict_single_img(filename)
        """
        if prediction_class[0] == 1:
            cv2.rectangle(face_img,(x,y),(x+w,y+h),(0,0,255),8)
            cv2.putText(face_img,'No mask',(x,y-25),cv2.FONT_HERSHEY_PLAIN,3,(0,0,255),4)
        else:
            cv2.rectangle(face_img,(x,y),(x+w,y+h),(0,255,0),4)
            cv2.putText(face_img,'Mask',(x,y-25),cv2.FONT_HERSHEY_PLAIN,3,(0,255,0),4)
           
    return face_img


def predict_single_img(roiImage):
    #img=image.load_img(imagefile, target_size=input_shape[:2])
    testimg=Image.fromarray(roiImage)
    testimg=testimg.resize((150,150),resample=0)
    testimg = image.img_to_array(testimg)
    testimg = np.expand_dims(testimg, axis=0)
    testimg = testimg/255
    prediction=model.predict(testimg)
    prediction_class=model.predict_classes(testimg)
    return (prediction,prediction_class[0])


def detect_video_mask(videofile):
    cap = cv2.VideoCapture(videofile)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter('demo_output/mask_capture_3.mp4', cv2.VideoWriter_fourcc(*'XVID'),25, (width, height))
    if cap.isOpened()== False: 
        print("Error opening the video file.")
    
    while True:
        ret, frame = cap.read() 
        frame = detectface(frame)
        writer.write(frame)
        cv2.imshow('Video Face Detection', frame)  
        if cv2.waitKey(1) & 0xFF == ord('q'): 
            break 
    cap.release()
    writer.release()
    cv2.destroyAllWindows()

input_shape = (150,150,3)
tt=cv2.imread('demodata/mask1.jpg')
detect_face=detectface(tt)
plt.imshow(detect_face)

detect_video_mask('demodata/video2.mp4')