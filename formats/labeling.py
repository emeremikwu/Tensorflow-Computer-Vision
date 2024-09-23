from dataclasses import dataclass, field
from typing import TypedDict
from pathlib import Path
from numpy import ndarray
import json


@dataclass
class DatasetEntry:
  """
  A data class to store information about a single dataset entry.

  Attributes:
      image_path (Path): The path to the image file.
      label_path (Path): The path to the ACTUAL label file
      target_label_directory (Path): The path where the label is to be stored (used for labeling).
      category (str): The category of the image (e.g., 'rock', 'paper', etc.).
      subset (str): The subset the image belongs to.
  """

  image_path: Path
  image_data: ndarray = field(default=None)

  label_target_path: Path = field(default=None)
  label_path: Path = field(default=None)
  label_data: ndarray = field(default=None)

  category: str = field(default=None)
  subset: str = field(default=None)


# starting to look like TypeScript
class Keypoint(TypedDict):
  label: str
  x: float
  y: float
  z: float | None


class KeypointMapping(TypedDict):
  left: list[Keypoint]
  right: list[Keypoint]


class LabelEntry(TypedDict):
  """
  A typed dictionary to store information about a single labeled entry. Used for writing to JSON.

  Attributes:
      image_path (str): The path to the image file.
      subset (str): The subset the image belongs to.
      category (str): The category of the image (e.g., 'rock', 'paper', etc.).
      keypoints (KeypointMapping): The keypoints of the image.
  """
  image_path: str
  subset: str | int
  category: str
  keypoints: KeypointMapping


if __name__ == "__main__":
  jsfile = json.load
