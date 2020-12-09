# -*- coding: utf-8 -*-
"""MaskDetection.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1lDfb32EUzTnnK_57UtD5l3Q_tEF3ITpg
"""

# Commented out IPython magic to ensure Python compatibility.
import cv2
import matplotlib.pyplot as plt
# %matplotlib inline
import numpy as np
import glob
import os

from keras.models import Sequential
from keras.layers import Dense, MaxPooling2D, Activation, Dropout,Flatten,Conv2D
from keras.utils.np_utils import to_categorical
from keras.preprocessing import image, image_dataset_from_directory
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split

normal_faces_img_directory='/content/drive/MyDrive/Colab Notebooks/MaskDetectionData/train/mask/'
masked_face_img_directory ='/content/drive/MyDrive/Colab Notebooks/MaskDetectionData/train/nomask/'
train_image_directory='/content/drive/MyDrive/Colab Notebooks/MaskDetectionData/train/'
validation_image_directory='/content/drive/MyDrive/Colab Notebooks/MaskDetectionData/validation/'
test_image_directory='/content/drive/MyDrive/Colab Notebooks/MaskDetectionData/test/'

def display(img):
  fig = plt.figure(figsize=(8,6))
  ax=fig.add_subplot(111)
  img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
  ax.imshow(img)

def loadImage(imageDirectory):
  image_data=[]
  datapath=os.path.join(imageDirectory,'*g')
  files=glob.glob(datapath)
  for file in files:
    img = cv2.imread(file)
    image_data.append(img)
  return image_data

def display_gray(img):
  fig = plt.figure(figsize=(8,6))
  ax=fig.add_subplot(111)
  img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
  ax.imshow(img,cmap='gray')

#normal_faces_images=loadImage(normal_faces_img_directory)

#single_image=normal_faces_images[1]
#display(single_image)

#masked_faces=loadImage(masked_face_img_directory)

from google.colab import drive
drive.mount('/content/drive')

#display(masked_faces[9])

def image_genarator():
  image_gen=ImageDataGenerator(
      zoom_range=0.2,
      shear_range=0.2,
      rescale=1/255,
      horizontal_flip=False
  )
  return image_gen

image_gen = image_genarator()
#display(image_gen.random_transform(masked_faces[9]))

train_img_gen=image_gen.flow_from_directory(train_image_directory)

train_img_gen.class_indices

input_shape = (150,150,3)

def prepare_model():
  model = Sequential()

  model.add(Conv2D(filters=32, kernel_size=(3,3),input_shape=input_shape, activation='relu',))
  model.add(MaxPooling2D(pool_size=(2, 2)))

  model.add(Conv2D(filters=32, kernel_size=(3,3),input_shape=input_shape, activation='relu',))
  model.add(MaxPooling2D(pool_size=(2, 2)))

  model.add(Conv2D(filters=64, kernel_size=(3,3),input_shape=input_shape, activation='relu',))
  model.add(MaxPooling2D(pool_size=(2, 2)))

  model.add(Flatten())

  model.add(Dense(64))
  model.add(Activation('relu'))

  model.add(Dropout(0.5))

  model.add(Dense(1))
  model.add(Activation('sigmoid'))

  model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
  return model

model=prepare_model()

model.summary()

def prepare_train_images():
  batch_size=16
  images_genarator=image_gen.flow_from_directory(train_image_directory,target_size=input_shape[:2],batch_size=batch_size,class_mode='binary')
  return images_genarator

def prepare_validation_images():
  batch_size=16
  images_genarator=image_gen.flow_from_directory(validation_image_directory,target_size=input_shape[:2],batch_size=batch_size,class_mode='binary')
  return images_genarator

train_img_gen=prepare_train_images()
validation_img_gen=prepare_validation_images()

results= model.fit_generator(train_img_gen,epochs=6,steps_per_epoch=200,validation_data=validation_img_gen,validation_steps=20)

model.save('/content/drive/MyDrive/Colab Notebooks/MaskDetectionData/maske_detection_model_phase1.h5')
results.history['accuracy']

def prepare_test_images():
  batch_size=16
  images_genarator=image_gen.flow_from_directory(test_image_directory,target_size=input_shape[:2],batch_size=batch_size,class_mode='binary')
  return images_genarator



def detectface(img):
  face_cascade=cv2.CascadeClassifier('/content/drive/MyDrive/Colab Notebooks/haarcasecades/haarcascades/haarcascade_frontalface_default.xml')
  face_img = img.copy()
  face_img = cv2.cvtColor(face_img,cv2.COLOR_BGR2GRAY)
  face_rects= face_cascade.detectMultiScale(face_img, scaleFactor=1.1,
        minNeighbors=5,minSize=(30, 30),flags=cv2.CASCADE_SCALE_IMAGE)
  for (x,y,w,h) in face_rects:
      cv2.rectangle(face_img,(x,y),(x+w,y+h),(0,255,0),4)
  return face_img

detect_face=detectface(tt)
plt.imshow(detect_face)

test_image_gen=prepare_test_images()

prediction=model.predict(test_image_gen,batch_size=16)

evaluation_result=model.evaluate(test_image_gen,batch_size=2,verbose=1)

testimg = image.load_img('/content/drive/MyDrive/Colab Notebooks/DemotestData/normal4.jpg', target_size=input_shape[:2])
testimg = image.img_to_array(testimg)
testimg = np.expand_dims(testimg, axis=0)
testimg = testimg/255
prediction=model.predict(testimg)
prediction_class=model.predict_classes(testimg)
print(prediction_class,prediction)

