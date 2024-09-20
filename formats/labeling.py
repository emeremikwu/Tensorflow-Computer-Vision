from dataclasses import dataclass, field
from typing import TypedDict
from pathlib import Path


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
  label_path: Path = field(default=None)

  image_width: int | float = field(default=None)
  image_height: int | float = field(default=None)
  target_label_directory: Path = field(default=None)
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


class JsonEntry(TypedDict):
  image_path: str
  subset: str | int
  category: str
  keypoints: KeypointMapping
