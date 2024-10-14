from unittests.dataset_generator import generate_temporary_dataset
from formats.labeling import JsonLabelEntry, KeypointMapping
from formats.landmark_lookup import LandmarkList
from pathlib import Path
import tempfile
import json
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from keras import models, layers, optimizers, losses, metrics, activations, regularizers
from sklearn.model_selection import train_test_split
import sys
print(tf.__version__)

# width, height
target_image_size = (256, 256)

# [ ] account for image resizing


def _json_to_tensor_base(json_path: Path) -> np.ndarray:
  json_data: JsonLabelEntry = json.load(open(json_path, 'r'))
  width, height = json_data.get('image_width'), json_data.get('image_height')

  # get keypoints
  mappings: KeypointMapping = json_data.get("keypoints")
  l_keypoints = [[m['x'], m['y']] for m in mappings.get('left')]
  r_keypoints = [[m['x'], m['y']] for m in mappings.get('right')]

  # scale keypoints to target image size and convert to numpy array
  # if len(l_keypoints):
  #   for i, (x, y) in enumerate(l_keypoints):
  #     l_keypoints[i] = [x * target_image_size[0] / width, y * target_image_size[1] / height]
  # else:
  #   l_keypoints = np.full((21, 2), -1)

  # if len(r_keypoints):
  #   for i, (x, y) in enumerate(l_keypoints):
  #     l_keypoints[i] = [x * target_image_size[0] / width, y * target_image_size[1] / height]
  # else:
  #   r_keypoints = np.full((21, 2), -1)

  # downscale keypoints to target image size and convert to numpy array
  l_keypoints = np.array(l_keypoints) * target_image_size / (width, height) if l_keypoints else np.full((21, 2), -1)
  r_keypoints = np.array(r_keypoints) * target_image_size / (width, height) if r_keypoints else np.full((21, 2), -1)

  # normalize keypoints to [0, 1]
  if not l_keypoints[0][0] == -1:
    l_keypoints = l_keypoints / target_image_size

  if not r_keypoints[0][0] == -1:
    r_keypoints = r_keypoints / target_image_size

  return (l_keypoints, r_keypoints)


def json_to_tensor_with_tf(json_path: Path) -> tf.Tensor:
  l_keypoints, r_keypoints = _json_to_tensor_base(json_path)

  l_tensor = tf.constant(l_keypoints, dtype=tf.float32)
  r_tensor = tf.constant(r_keypoints, dtype=tf.float32)
  return tf.concat([l_tensor, r_tensor], axis=0)


def json_to_tensor_without_tf(json_path: Path) -> np.ndarray:
  l_keypoints, r_keypoints = _json_to_tensor_base(json_path)

  return np.concatenate([l_keypoints, r_keypoints], axis=0)


def image_to_tensor_without_tf(image_path: Path) -> np.ndarray:
  image = cv2.imread(str(image_path))
  image = cv2.resize(image, target_image_size)
  image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
  image = image / 255.0  # normalize to [0, 1]
  return image


def image_to_tensor_with_tf(image_path: Path) -> tf.Tensor:
  # using tensorflow
  image = tf.io.read_file(str(image_path))
  image = tf.image.decode_jpeg(image, channels=3)
  image = tf.image.resize(image, target_image_size)
  image = tf.image.convert_image_dtype(image, tf.float32)  # normalize to [0, 1]
  return image


def masked_mse(y_true, y_pred):
    # Assuming missing keypoints are represented by [-1, -1, -1]
  mask = tf.not_equal(y_true, -1.0)  # Mask to ignore missing keypoints
  mask = tf.cast(mask, dtype=tf.float32)  # Convert to float32 for multiplication

  # Calculate the MSE but only for valid keypoints
  mse = tf.square(y_true - y_pred)
  masked_mse = tf.reduce_sum(mse * mask) / tf.reduce_sum(mask)  # Normalize by number of valid keypoints

  return masked_mse


def display_stats(history: tf.keras.callbacks.History):
  print(f"Training accuracy: {history.history['accuracy'][-1]}")
  print(f"Validation accuracy: {history.history['val_accuracy'][-1]}")
  print(f"Training loss: {history.history['loss'][-1]}")
  print(f"Validation loss: {history.history['val_loss'][-1]}")

  # Plot training and validation loss
  plt.plot(history.history['loss'], label='Training Loss')
  plt.plot(history.history['val_loss'], label='Validation Loss')
  plt.title('Model Loss over Epochs')
  plt.xlabel('Epochs')
  plt.ylabel('Loss')
  plt.legend()
  plt.show()

# Plot accuracy if available
  if 'accuracy' in history.history:
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()


def save_model(model: models.Model, path: Path):
  model.save(path)
  print(f"Model saved to {path}")


def load_model(path: Path) -> models.Model:
  return models.load_model(path)


def test_image_against_model(model: models.Model, image_path: Path, unscale_keypoints: bool = True):
  # load image in bgr format and original size
  image_orig = cv2.imread(str(image_path))

  # load image in rgb format and target size
  image = image_to_tensor_with_tf(image_path)
  # converts shape from (256, 256, 3) is (1, 256, 256, 3)
  # (batch, height, width, channels)
  image = tf.expand_dims(image, axis=0)

  # predict keypoints
  predicted_keypoints = model.predict(image)

  # remove batch dimension
  predicted_keypoints = np.squeeze(predicted_keypoints)

  # unscale keypoints
  if unscale_keypoints:
    predicted_keypoints = predicted_keypoints / target_image_size * image_orig.shape[:2]

  return predicted_keypoints


if __name__ == "__main__":

  use_tf = False

  load_image = image_to_tensor_with_tf if use_tf else image_to_tensor_without_tf
  load_json = json_to_tensor_with_tf if use_tf else json_to_tensor_without_tf

  system_temp_dir = Path(tempfile.gettempdir())
  # print working directory
  dataset_path = next(system_temp_dir.glob("dataset_*"), None) or generate_temporary_dataset(
      str(Path.cwd() / "unittests/images"), delete=False)[0]
  print(f"Dataset path: {dataset_path}")

  json_files = list(dataset_path.rglob("*.json"))
  image_files = list(dataset_path.rglob("*.jpg"))

  if not json_files:
    print("No json files found, running labeler...")
    from dataset_tools import loader, labeler
    _loader = loader.DatasetLoader(dataset_path, load_labels=False, load_label_data=False)
    labeler.DatasetLabeler(_loader).process_images()
    json_files = list(dataset_path.rglob("*.json"))

  assert len(json_files) == len(image_files), "Number of json files and image files do not match"

  print(f"loaded {len(json_files)} json files in dataset path")

  # load images and labels
  # needs to be either a list of numpy arrays to work with train_test_split
  images = np.array([load_image(path) for path in image_files])
  labels = np.array([load_json(path) for path in json_files])

  assert len(images) == len(labels), "Number of images and labels do not match"

  x_train, x_test, y_train, y_test = train_test_split(images, labels, test_size=0.2)

  model = models.Sequential()

  # input layer
  model.add(layers.Input(shape=target_image_size + (3,)))
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
      loss=masked_mse,
      metrics=['accuracy']
  )

  model.summary()

  history = model.fit(x_train, y_train, epochs=10, batch_size=32, validation_data=(x_test, y_test))
  breakpoint()
  prediction = test_image_against_model(model, sys.argv[6])
  breakpoint()
  # display_stats(history)
