import logging
import cv2

from numpy import ndarray
from dataclasses import dataclass, field
import mediapipe as mp
from mediapipe.tasks.python.components.containers import Landmark

from utils import results_to_json, write_json_to_file
from loader import DatasetLoader
from formats.labeling import DatasetEntry

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands


@dataclass
class Datasetlabeler:
  _dataset_loader: DatasetLoader = None
  overwrite_existing: bool = False
  _ready = False
  _error_list = []
  _hand_detector = field(init=False, default=None)

  # initialize mediapipe hands
  def __post_init__(self):
    self._check_if_ready()

    self._hand_detector = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.5)

  @property
  def dataset(self, dataset):
    self._dataset_loader = dataset
    self._check_if_ready
    return self

  @dataset.setter
  def dataset(self, dataset):
    self._dataset_loader = dataset

  @property
  def is_ready(self):
    self._check_if_ready()
    self._ready = len(self._error_list) == 0

    if not self._ready:
      for error in self._error_list:
        logging.error(error)

    return self._ready

  def process_images(self):

    if not self.is_ready:
      return

    logging.info("Processing images...")

    for entry in self._dataset_loader.get_iterator():

      logging.debug(f'Processing image: {entry.image_path}, {entry.image_data.shape}')

      # Convert the BGR image to RGB before processing.
      image_copy = cv2.flip(entry.image_data.copy(), 1)
      results = self._hand_detector.process(cv2.cvtColor(image_copy, cv2.COLOR_BGR2RGB))

      # Flip the x-coordinates of the landmarks
      for hand_landmarks in results.multi_hand_landmarks:
        for landmark in hand_landmarks.landmark:
          landmark.x = 1 - landmark.x

      json_data = results_to_json(entry, entry.image_data, results)
      write_json_to_file(json_data, entry.label_target_path)  # [ ] - error handling

  def label(self, dataset):
    for item in dataset:
      item['label'] = self._dataset_loader.get_label(item['id'])
    return dataset

  def _check_if_ready(self):
    self._error_list = []

    if not self._dataset_loader or not self._dataset_loader.initialized:
      self._error_list.append(ValueError("DatasetLoader instance not set or not initialized"))  # Raise the exception
      return

    if not self._dataset_loader.load_images:
      self._error_list.append(ValueError("DatasetLoader instance not set to load images"))

    if not self._dataset_loader.load_image_data:
      self._error_list.append(ValueError("DatasetLoader instance not set to load image data"))

    if not self._dataset_loader.lazy_load:
      logging.warning("DatasetLoader instance not set to lazy load images, this may cause memory issues for large datasets")
