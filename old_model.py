import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt
import math

print("Starting Machine Learning")
print("Tensorflow ",tf.VERSION)
print("Keras ",tf.keras.__version__)

def sigmoid(x):
      return 1 / (1 + math.exp(-x))

norm = np.vectorize(sigmoid)

fig = norm(np.random.rand(32,20))

plt.figure()
plt.imshow(fig)
plt.colorbar()
plt.grid(False)

model = tf.keras.Sequential()
model.add(layers.Dense(2304, activation='relu'))
model.add(layers.Dense(1600, activation='relu'))
model.add(layers.Dense(1600, activation='relu'))
model.add(layers.Dense(2304, activation='softmax' ))

model.compile(optimizer=tf.train.AdamOptimizer(0.001),
  `                    loss='categorical_crossentropy',
                                    metrics=['accuracy'])

model.build(input_

print("The Machine has learned")
plt.show()
