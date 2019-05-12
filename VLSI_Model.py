import tensorflow as tf
from   tensorflow.keras import layers
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import math
from math import floor
import h5py
from plot_helper import draw_circ 
from plot_helper import draw_group

train_images = []
train_centers = []
train_boxes = []

test_images = []
test_centers = []
test_boxes = []

print("Starting Machine Learning")
print("VLSI Defect detection System, by Ali Tariq")
print("data reference samyzaf.com/ML/opens/opens.html")
print("Tensorflow ",tf.VERSION)
print("Keras ",tf.keras.__version__)

features = [(i,j) for i in range(48) for j in range(48)]

classes = range(64)

f = h5py.File('data/opens1.h5', 'r')

print("loading training data")

for i in range(10000):
    box_key = 'open_' + str(i)
    img_key = 'img_' + str(i)
    center_key = 'center_' + str(i)

    box = np.array(f.get(box_key))
    train_boxes.append(box)

    img = np.array(f.get(img_key))/255
    train_images.append(np.logical_not(img))

    c = np.array(f.get(center_key))
    train_centers.append(floor((c[0]*48 + c[1])*63/2304))

print("loading testing data")


for i in range(10001,20000):
    box_key = 'open_' + str(i)
    img_key = 'img_' + str(i)
    center_key = 'center_' + str(i)

    box = np.array(f.get(box_key))
    test_boxes.append(box)

    img = np.array(f.get(img_key))/255
    test_images.append(np.logical_not(img))

    c = np.array(f.get(center_key))
    test_centers.append(floor((c[0]*48 + c[1])*63/2304))


model = tf.keras.Sequential()
model.add(layers.Flatten(input_shape =(48,48)))
model.add(layers.Dense((2304 + 64)/2, activation='linear'))
model.add(layers.Dense(592,activation = 'linear'))
model.add(layers.Dense(64, activation='softmax'))

model.compile(optimizer=tf.train.AdamOptimizer(0.001),loss='sparse_categorical_crossentropy',metrics=['accuracy'])

model.fit(np.array(train_images),np.array(train_centers),epochs=5)

print("running test")

test_loss, test_acc = model.evaluate(np.array(test_images), np.array(test_centers))

print('Test accuracy:', test_acc)


print("The Machine has learned")
