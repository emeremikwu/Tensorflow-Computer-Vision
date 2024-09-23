# TODO: cleanup
from mediapipe.framework.formats import landmark_pb2
from loader import DatasetEntry
from numpy import ndarray
import json

from formats.labeling import LabelEntry, Keypoint
from formats.landmark_lookup import LandmarkDictionary

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


def results_to_json(dataset_entry: DatasetEntry, image: ndarray, results) -> LabelEntry:
  keypoints = landmark_to_dict(image, results)
  image_path = dataset_entry.image_path
  output_path = dataset_entry.label_target_path

  entry: LabelEntry = {
      "image_path": image_path,
      "subset": dataset_entry.subset,
      "category": dataset_entry.category,
      "keypoints": keypoints
  }

  return entry


def write_json_to_file(data: json, destination: str) -> None:
  with open(str(destination), 'w') as output_file:
    json.dump(data, output_file, indent=4)


def read_json_from_file(source: str) -> json:
  with open(str(source), 'r') as input_file:
    data = json.load(input_file)
    return data
