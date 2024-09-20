import logging
import cv2
from utils import landmark_to_dict
import mediapipe as mp
from mediapipe.tasks.python.components.containers import Landmark
from mediapipe.framework.formats import landmark_pb2
from numpy import ndarray

from formats.labeling import DatasetEntry


mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

# [ ] - implement
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


def process_image(image_file: ndarray):
  # For static images:
  with mp_hands.Hands(
          static_image_mode=True,
          max_num_hands=2,
          min_detection_confidence=0.5) as hands:
    
    logging.debug(f"Processing image: {image_file}")
    # Convert the BGR image to RGB before processing.
    results = hands.process(cv2.cvtColor(image_file, cv2.COLOR_BGR2RGB))

    # Print handedness and draw hand landmarks on the image.
    print('Handedness:', results.multi_handedness)
    # if not results.multi_hand_landmarks:
    #   continue
    image_height, image_width, _ = image_file.shape
    annotated_image = cv2.flip(image_file.copy(), 1)
    keypointDict = landmark_to_dict(image_file, results)

    for hand_landmarks in results.multi_hand_landmarks:
      # print('hand_landmarks:', hand_landmarks)
      print(
          f'Index finger tip coordinates: (',
          f'{hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * image_width}, '
          f'{hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * image_height})'
      )

      # flip landmarks to match the original image
      flipped_landmarks = hand_landmarks
      for landmark in flipped_landmarks.landmark:
        landmark.x = 1 - landmark.x

      mp_drawing.draw_landmarks(
          annotated_image,
          flipped_landmarks,  # hand_landmarks,
          mp_hands.HAND_CONNECTIONS,
          mp_drawing_styles.get_default_hand_landmarks_style(),
          mp_drawing_styles.get_default_hand_connections_style())

def create_image_label(dataset_entry: DatasetEntry, results): 
