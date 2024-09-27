# fmt: off
import sys
from itertools import repeat
sys.path.append("..")
# from dataset_tools.utils import landmark_to_dict
import cv2
import pathlib
import numpy as np
import mediapipe as mp
from mediapipe import Image, solutions
from mediapipe.framework.formats import landmark_pb2
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
# fmt: on


mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

MARGIN = 10  # pixels
FONT_SIZE = 1
FONT_THICKNESS = 1
HANDEDNESS_TEXT_COLOR = (88, 205, 54)  # vibrant green


def draw_landmarks(image, landmarks):
  for landmark in landmarks.landmark:
    x = int(landmark.x * image.shape[1])
    y = int(landmark.y * image.shape[0])
    cv2.circle(image, (x, y), 5, (0, 255, 0), -1)
  return image


def draw_landmarks_on_image(rgb_image, detection_result):
  hand_landmarks_list = detection_result.hand_landmarks
  handedness_list = detection_result.handedness
  annotated_image = np.copy(rgb_image)

  # Loop through the detected hands to visualize.
  for idx in range(len(hand_landmarks_list)):
    hand_landmarks = hand_landmarks_list[idx]
    handedness = handedness_list[idx]

    # Draw the hand landmarks.
    hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
    hand_landmarks_proto.landmark.extend([
        landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in hand_landmarks
    ])

    solutions.drawing_utils.draw_landmarks(
        annotated_image,
        hand_landmarks_proto,
        solutions.hands.HAND_CONNECTIONS,
        solutions.drawing_styles.get_default_hand_landmarks_style(),
        solutions.drawing_styles.get_default_hand_connections_style())

    # Get the top left corner of the detected hand's bounding box.
    # height, width, _ = annotated_image.shape
    # y_coordinates = [landmark.x for landmark in hand_landmarks]
    # x_coordinates = [landmark.y for landmark in hand_landmarks]
    # text_x = int(min(x_coordinates) * width)
    # text_y = int(min(y_coordinates) * height) - MARGIN

    # # Draw handedness (left or right hand) on the image.
    # cv2.putText(annotated_image, f"{handedness[0].category_name}",
    #             (text_x, text_y), cv2.FONT_HERSHEY_DUPLEX,
    #             FONT_SIZE, HANDEDNESS_TEXT_COLOR, FONT_THICKNESS, cv2.LINE_AA)

  return annotated_image


def impl1():
  img_path = sys.argv[2]
  image_file_cv2 = cv2.imread(img_path)
  image_file_mp = Image.create_from_file(img_path)
  annotated_image = image_file_cv2.copy()
  task_file = "testing/hand_landmarker.task"

  # cv2.imshow("Image", image_file_cv2)
  # cv2.waitKey(0)

  # STEP 2: Create an HandLandmarker object.
  base_options = python.BaseOptions(model_asset_path=task_file)
  options = vision.HandLandmarkerOptions(base_options=base_options,
                                         num_hands=2)
  detector = vision.HandLandmarker.create_from_options(options)
  detection_results = detector.detect(image_file_mp)

  # STEP 3: Draw landmarks on the image.
  annotated_image = draw_landmarks_on_image(annotated_image, detection_results)
  cv2.imshow("Annotated Image", annotated_image)
  cv2.waitKey(0)


def impl2():
  # For static images:
  # ["/tmp/dataset_s58dz28v/NQnywzqvbS/paper/images/both_hands.jpg", sys.argv[5]]
  # IMAGE_FILES = list(repeat("/tmp/dataset_crs0efao/tSzhwGMpSa/paper/images/both_hands.jpg", 10))

  IMAGE_FILES = [str(img) for img in pathlib.Path("/tmp/dataset_crs0efao/tSzhwGMpSa/").rglob("*.jpg")]
  with mp_hands.Hands(
          static_image_mode=True,
          max_num_hands=2,
          min_detection_confidence=0.5) as hands:
    for idx, file in enumerate(IMAGE_FILES):
      # Read an image, flip it around y-axis for correct handedness output (see
      # above).
      image = cv2.flip(cv2.imread(file), 1)
      # image = cv2.imread(file)
      # Convert the BGR image to RGB before processing.
      results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

      # Print handedness and draw hand landmarks on the image.
      print('Handedness:', results.multi_handedness)
      if not results.multi_hand_landmarks:
        continue
      image_height, image_width, _ = image.shape
      annotated_image = cv2.flip(image.copy(), 1)
      # keypointDict = landmark_to_dict(image, results)

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
      # cv2.imwrite(
      #     '/tmp/annotated_image' + str(idx) + '.png', cv2.flip(annotated_image, 1))
      # Draw hand world landmarks.
      if not results.multi_hand_world_landmarks:
        continue

      # cv2.imshow('MediaPipe Hands', annotated_image)
      # cv2.waitKey(0)
      # for hand_world_landmarks in results.multi_hand_world_landmarks:
      #   mp_drawing.plot_landmarks(
      #       hand_world_landmarks, mp_hands.HAND_CONNECTIONS, azimuth=5)


if __name__ == "__main__":
  impl2()
