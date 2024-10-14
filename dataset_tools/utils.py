# TODO: cleanup
from mediapipe.framework.formats import landmark_pb2
from formats.labeling import DatasetEntry
from numpy import ndarray
import json
from pathlib import Path
from formats.labeling import JsonLabelEntry, Keypoint
from formats.landmark_lookup import LandmarkDictionary
import cv2
# [ ] - delete
exp = {
    "image_path": "path/relative/to/dataset",
    "subset": 1,  # can also be a string
    "category": "rock",
    "keypoints": {
        "left": [
            {
                "label": "thumb",
                "x": 0.1,
                "y": 0.2,
                "z": 0.3
            }
        ],

        "right": [
            {
                "label": "thumb",
                "x": 0.1,
                "y": 0.2,
                "z": 0.3
            }
        ]
    }
}


def landmark_to_dict(image: ndarray, results):

  image_width, image_height, _ = image.shape

  keypoints: Keypoint = {
      "left": [],
      "right": [],
  }

  if not results.multi_hand_landmarks:
    return keypoints

  for i, hand in enumerate(results.multi_handedness):
    handedness = str(hand.classification[0].label).lower()

    for j, landmark in enumerate(results.multi_hand_landmarks[i].landmark):
      keypoint: Keypoint = {
          "label": LandmarkDictionary[j],
          "x": landmark.x * image_width,
          "y": landmark.y * image_height,
          "z": landmark.z
      }

      keypoints[handedness].append(keypoint)

  return keypoints


def results_to_json(dataset_entry: DatasetEntry, image: ndarray, results) -> JsonLabelEntry:
  keypoints = landmark_to_dict(image, results)
  image_path = str(dataset_entry.image_path)

  width, height, _ = image.shape

  entry: JsonLabelEntry = {
      "image_path": image_path,
      "image_width": width,
      "image_height": height,
      "subset": dataset_entry.subset,
      "category": dataset_entry.category,
      "keypoints": keypoints
  }

  return entry


def write_json_to_file(data: json, destination: str | Path) -> None:
  if (isinstance(destination, Path)):
    destination = str(destination)

  with open(str(destination), 'w') as output_file:
    json.dump(data, output_file, indent=4)


def read_json_from_file(source: str) -> json:
  with open(str(source), 'r') as input_file:
    data = json.load(input_file)
    return data


# normalizedlandmark import


def draw_landmarks_on_image(annotated_image: ndarray, results) -> ndarray:
  image_width, image_height, _ = annotated_image.shape

  landmarks = landmark_pb2.NormalizedLandmarkList()

  for i, hand in enumerate(results.multi_handedness):
    handedness = str(hand.classification[0].label).lower()

    for j, landmark in enumerate(results.multi_hand_landmarks[i].landmark):
      cx, cy = int(landmark.x * image_width), int(landmark.y * image_height)
      cv2.circle(annotated_image, (cx, cy), 5, (0, 255, 0), -1)

  return annotated_image
