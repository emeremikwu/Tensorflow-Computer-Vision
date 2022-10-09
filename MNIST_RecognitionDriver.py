# AUTHOR(S):		Mauricio Cerda, Emmanuel Meremikwu
# FILENAME:  		MNIST_Example.py
# SPECIFICATION: 	Simple computer vision project to recognize and catalog certain objects in a given image
# FOR:       		CS 3368 Introduction to Artificial Intelligence Section 001

##### Loading Libraries & Setup #####
# NumPy - mathematical functions on multi-dimensional arrays and matrices
import numpy as np

# Matplotlib - plotting library to create graphs and charts
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# Tensorflow - deep learning software installed and to be used
import tensorflow as tf

# MNIST - digital database of handwritten digits
from tensorflow.keras.datasets import mnist

# Keras - Python open-source neural-network library
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical

from IPython.display import Image
from IPython.core.display import display, HTML

# scikit-learn
import sklearn.preprocessing as preprocessing

from tensorflow.keras.preprocessing import image

# cv2
import cv2

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

model = tf.keras.models.Sequential([tf.keras.layers.Flatten(input_shape=(28,28)),tf.keras.layers.Dense(128, activation='relu'),tf.keras.layers.Dropout(0.2),tf.keras.layers.Dense(10)])

loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

model.compile(optimizer='adam', loss=loss_fn, metrics=['accuracy'])

model.fit(x_train, y_train, epochs=10)



'''

##### Preparing Training Data #####

# From MNIST, importing, training, and testing images. Keeping original images to display some digits initially and during testing.
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

# Looking at the shape of the data. NOTE: There are 60,000 images and the size of image is 28x28 pixels, represented as 28 by 28 matrix by numbers
#train_images.shape
assert x_train.shape == (60000, 28, 28)
assert x_test.shape == (10000, 28, 28)
assert y_train.shape == (60000,)
assert y_test.shape == (10000,)

# Before normalizing the training data, grab some images for display.
train_images_display = train_images[:5]

# Normalizing and flattening training images data. Better format to use for Neural Network training.
# Normalize the images.
train_images = (train_images / 255) - 0.5

# Flatten the images - changing the dimension of the array from 28x28 to 1x784.
train_images = train_images.reshape((-1,784))

print('Training array has a shape: ' + str(train_images.shape))
print('Each element (image) has a shape: ' + str(train_images[0].shape))

##### Printing Some Digits #####

# By using matplotlib

# Displaying images - this takes MUCH memory.
f = plt.figure(figsize=(10,5))
columns = 5
images = train_images_display[:20]
for i, image in enumerate(images):
    plt.subplot(len(images) / columns + 1, columns, i + 1)
    # imshow displays array-like images
    plt.imshow(image)

plt.show()
f.clear()
plt.close(f)
'''