from keras import models, layers, optimizers, losses, metrics, activations, callbacks
import numpy as np
import tensorflow as tf
from formats.landmark_lookup import LandmarkList

import logging


class LandmarkModel():

  def __init__(self, input_shape: tuple, output_shape: tuple):
    self.input_shape = input_shape
    self.output_shape = output_shape
    self.model = self.build_model()
    self.recent_history: callbacks.History = None

  @staticmethod
  def from_saved_model(model_path: str):
    model: models.Sequential = models.load_model(model_path)
    input_shape = model.input_shape[1:]
    output_shape = model.output_shape[1:]
    return LandmarkModel(input_shape, output_shape)

  def build_model(self) -> models.Sequential:

    model = models.Sequential()
    model.add(layers.Input(shape=self.input_shape))
    # model.add(layers.Masking(mask_value=-1.0))  # Mask missing keypoints

    # Convolutional layers
    model.add(layers.Conv2D(32, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))

    # Flatten the features extracted by convolutional layers
    model.add(layers.Flatten())

    # Fully connected layers (Dense layers)
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.Dropout(0.5))

    model.add(layers.Dense(2 * len(LandmarkList) * 2), activations.sigmoid)
    model.add(layers.Reshape((2 * len(LandmarkList), 2)))

    model.compile(
        optimizer='adam',
        # loss='mean_squared_error',
        loss=self.masked_mse,
        metrics=['accuracy']
    )
    return model

  def masked_mse(self, y_true, y_pred):
    mask = tf.not_equal(y_true, -1.0)  # Mask to ignore missing keypoints
    mask = tf.cast(mask, dtype=tf.float32)  # Convert to float32 for multiplication

    # Calculate the MSE but only for valid keypoints
    mse = tf.square(y_true - y_pred)
    masked_mse = tf.reduce_sum(mse * mask) / tf.reduce_sum(mask)  # Normalize by number of valid keypoints

    return masked_mse

  def train(self, x_train, y_train, x_test, y_test, epochs=10, batch_size=32) -> callbacks.History:
    self.recent_history = self.model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(x_test, y_test))
    return self.recent_history

  # [ ] - move to utils or implement base class
  def print_metrics(self, history: callbacks.History = None) -> None:

    local_history: callbacks.History = history or self.recent_history
    logging.info(f"Training accuracy: {local_history.history['accuracy'][-1]}")
    logging.info(f"Validation accuracy: {local_history.history['val_accuracy'][-1]}")
    logging.info(f"Training loss: {local_history.history['loss'][-1]}")
    logging.info(f"Validation loss: {local_history.history['val_loss'][-1]}")

  def save(self, model_path: str) -> None:
    self.model.save(model_path)
    logging.info(f"Model saved to {model_path}")
