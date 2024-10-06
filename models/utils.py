

import numpy as np
from keras import models
from tensorflow import expand_dims
from cv2 import imread, resize, cvtColor, COLOR_BGR2RGB
from formats.labeling import JsonLabelEntry, KeypointMapping
import json

DEFAULT_IMAGE_SIZE = (256, 256)


def load_json(json_path: str, target_size: tuple = DEFAULT_IMAGE_SIZE) -> np.ndarray:
  # load json  data
  json_data: JsonLabelEntry = json.load(open(json_path, 'r'))
  width, height = json_data.get('image_width'), json_data.get('image_height')

  # get keypoints from data and convert to numpy array
  mappings: KeypointMapping = json_data.get("keypoints")
  l_keypoints = [[m['x'], m['y']] for m in mappings.get('left')]
  r_keypoints = [[m['x'], m['y']] for m in mappings.get('right')]

  # downscale keypoints to target image size and convert to numpy array
  l_keypoints = np.array(l_keypoints) * target_size / (width, height) if l_keypoints else np.full((21, 2), -1)
  r_keypoints = np.array(r_keypoints) * target_size / (width, height) if r_keypoints else np.full((21, 2), -1)

  # normalize keypoints to [0, 1]
  if not l_keypoints[0][0] == -1:
    l_keypoints = l_keypoints / target_size

  if not r_keypoints[0][0] == -1:
    r_keypoints = r_keypoints / target_size

  return np.concatenate([l_keypoints, r_keypoints], axis=0)


def load_image(image_path: str, target_size: tuple = DEFAULT_IMAGE_SIZE) -> np.ndarray:
  # load image from path
  image = imread(image_path)
  image = resize(image, target_size)
  image = cvtColor(image, COLOR_BGR2RGB)

  # normalize image to [0, 1]
  image = image / 255.0
  return image


def test_image_against_model(
        model: models.Model,
        image_path: str,
        unscale_keypoints: bool = True,
        target_image_size: tuple = DEFAULT_IMAGE_SIZE) -> np.ndarray:
  # load image in bgr format and original size
  image_orig = imread(image_path)

  # load image in rgb format and target size
  image = load_image(image_path)
  # converts shape from (256, 256, 3) is (1, 256, 256, 3)
  # (batch, height, width, channels)
  image = expand_dims(image, axis=0)

  # predict keypoints
  predicted_keypoints = model.predict(image)

  # remove batch dimension
  predicted_keypoints = np.squeeze(predicted_keypoints)

  # unscale keypoints
  if unscale_keypoints:
    predicted_keypoints = predicted_keypoints / target_image_size * image_orig.shape[:2]

  return predicted_keypoints
