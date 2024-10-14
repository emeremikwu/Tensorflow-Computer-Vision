import argparse


description = """
This application provides two main functionalities:
1. Labeling images using MediaPipe for keypoint extraction and storing the keypoints in a structured format.
2. Training a model on a dataset that has already been labeled, with the aim of recognizing hand gestures based on the labeled keypoints.
"""


def configure_parser():
  parser = argparse.ArgumentParser(
      prog="gesture_recognition",
      description=description
  )

  # parser.add_argument(
  #     "path",
  #     help="path to the dataset directory or model file",
  #     nargs="?",
  # )

  parser.add_argument(
      "-v", "--verbose",
      help="increase output verbosity",
      action="store_true"
  )

  # -------- train subparser --------
  train_subparser: argparse.ArgumentParser = parser.add_subparsers(
      "train",
      help="train a model on a dataset that has already been labeled",
  )

  # train_subparser.add_argument(
  #     "-m", "--model",
  #     help="path to the model file",
  #     type=str,
  # )

  train_subparser.add_argument(
      "dataset",
      help="path to the dataset directory",
      type=str,
      nargs="?",
  )

  train_subparser.add_argument(
      "-m", "--model",
      help="path to the model file",
      type=str,
  )

  train_subparser.add_argument(
      "-o", "--output",
      help="model output directory",
  )

  # -------- label subparser --------
  label_subparser: argparse.ArgumentParser = parser.add_subparsers(
      "label",
      help="label images using MediaPipe for keypoint extraction",
      action="store_true"
  )

  label_subparser.add_argument(
      "-d", "--dataset",
      help="path to the dataset directory",
      type=str,
  )

  return parser
