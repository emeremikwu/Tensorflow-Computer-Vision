from pathlib import Path
import mediapipe as mp
from mediapipe.tasks.python.components.containers import Landmark
from mediapipe.framework.formats import landmark_pb2
import json

from mediapipe.python import c

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode


def approximate_palm(landmarks: list[mp.datastructures.Landmark]) -> list[mp.datastructures.Landmark]:
  # Get the landmarks for the wrist and the palm:
  wrist_landmark = landmarks[0]
  palm_landmark = landmarks[-1]

  # Calculate the approximate palm landmarks:
  approximate_palm_landmarks = [
      wrist_landmark,
      palm_landmark
  ]

  return approximate_palm_landmarks


def create_image_label(
        input_image_path: str | Path,
        output_label_path: str | Path):

  if isinstance(input_image_path, str):
    input_image_path = Path(input_image_path)

  if isinstance(output_label_path, str):
    output_label_path = Path(output_label_path)


if __name__ == "__main__":
  # Create a hand landmarker instance with the image mode:
  options = HandLandmarkerOptions(
      base_options=BaseOptions(model_asset_path='/testing/hand_landmarker.task'),
      running_mode=VisionRunningMode.IMAGE)

  landmarker = HandLandmarker.create_from_options(options)
