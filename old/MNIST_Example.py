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

# Defining functions to load particular data sets -> MNIST 0-9 and Kaggle A-Z
def load_mnist_dataset():
   # loading data from tensorflow framework
   ((trainData, trainLabels), (testData, testLabels)) = mnist.load_data()

   # Stacking train data and test data to form single array named data
   data = np.vstack([trainData, testData])

   # Vertical stacking labels of train and test set
   labels = np.hstack([trainLabels, testLabels])

   # return a 2-tuple of the MNIST data and labels
   return (data, labels)

def load_az_dataset(datasetPath):
   # List for storing data
   data = []

   # List for storing labels
   labels = []

   for row in open(datasetPath): #Openfile and start reading each row
     #Split the row at every comma
     row = row.split(",")

     #row[0] contains label
     label = int(row[0])

     #Other all columns contains pixel values make a separate array for that
     image = np.array([int(x) for x in row[1:]], dtype="uint8")

     #Reshaping image to 28 x 28 pixels
     image = image.reshape((28, 28))

     #append image to data
     data.append(image)

     #append label to labels
     labels.append(label)

   #Converting data to numpy array of type float32
   data = np.array(data, dtype='float32')

   #Converting labels to type int
   labels = np.array(labels, dtype="int")

   return (data, labels)

# Calling functions
(digitsData, digitsLabels) = load_mnist_dataset()
(azData, azLabels) = load_az_dataset('/Users/mauriciocerda/A_ZHandwrittenData.csv')

# Combining both datasets to feed into model
# NOTE: The MNIST dataset occupies the labels 0-9, so let's add 10 to every A-Z label to ensure the A-Z characters are not incorrectly labeled.

azLabels += 10

# stack the A-Z data and labels with the MNIST dights data and labels

data = np.vstack([azData, digitsData])
labels = np.hstack([azLabels, digitsLabels])

# Each image in the A-Z and MNIST digits datasets are 28x28 pixels;
# However, the model architecture we're using is designed for 32x32 images,
# So we need to resize them to 32x32

data = [cv2.resize(image, (32,32)) for image in data]
data = np.array(data, dtype="float32")

# To resize it further, now adding a channel dimension to every image in the dataset and scale the pixel intensities of the images, from [0,255] down to [0,1]
data = np.expand_dims(data, axis = -1)
data /= 255.0

# NOW, convert labels from <integer> to <vector> for ease in model fitting and see the count the weights of each character in the dataset and also count the
# classweights for each label.
le = preprocessing.LabelBinarizer()
labels = le.fit_transform(labels)

counts = labels.sum(axis=0)

# account for skew in the labeled data
classTotals = labels.sum(axis=0)
classWeight = {}

# loop over all classes and calculate the class weight
for i in range(0, len(classTotals)):
  classWeight[i] = classTotals.max() / classTotals[i]


# Finally, we can improve results of our ResNet classifier by augmenting the input data for training using ImageDataGenerator.
# Constructing the image generator for data augmentation

aug = image.ImageDataGenerator(rotation_range=10,zoom_range=0.05,width_shift_range=0.1,height_shift_range=0.1,shear_range=0.15,horizontal_flip=False,fill_mode="nearest")


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

'''
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

model = tf.keras.models.Sequential([tf.keras.layers.Flatten(input_shape=(28,28)),tf.keras.layers.Dense(128, activation='relu'),tf.keras.layers.Dropout(0.2),tf.keras.layers.Dense(10)])

loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

model.compile(optimizer='adam', loss=loss_fn, metrics=['accuracy'])

model.fit(x_train, y_train, epochs=10)
'''

