import cv2
import sys
import logging
import os
import json
from dataclasses import dataclass, field
from pathlib import Path  # Experimenting with Path and os.path for learning experience
from typing import Iterator

from formats.labeling import DatasetEntry


DirectoryConstraints = {
    "expected": ['rock', 'paper', 'scissors', 'uncategorized'],
    "ignored": [".DS_Store"]
}


@dataclass
class DatasetLoader:
  """
  A class responsible for loading and validating a structured dataset.

  Attributes:
      root_path (str): The root path of the dataset.

      structured (bool): Whether the dataset is structured or not.
      skip_validation (bool): Flag to skip validation of the dataset structure.
      skip_non_uniform_directory (bool): Whether to skip directories with non-uniform file extensions.
      ignore_random_files (bool): Whether to ignore random files in the dataset.
      lazy_load (bool): Whether to lazy load the dataset

      load_images (bool): Whether to load images.
      load_image_data (bool): Whether to load image data into memory.
      load_labels (bool): Whether to load labels.
      load_label_data (bool): Whether to load label data.

  """
  root_path: str

  structured = True   # [ ] - implement or remove. I don't see a reason to use a non-structured dataset
  skip_validation = False
  skip_non_uniform_directory = True
  skip_missing_labels = True
  ignore_random_files = True
  lazy_load = True

  # [ ] Implement attributes load_images and load_labels
  load_images = True
  load_image_data = True
  load_labels = True
  load_label_data = True

  initialized: bool = field(default=False, init=False)
  # supress_missing_labels_warning: bool = field(default=True, init=False)

  _validated: bool = field(default=False, init=False)
  _dataset: list = field(default=None, init=False)  # Holds list of DatasetEntry objects if eager-loaded

  def __post_init__(self) -> None:
    """
    Initialization hook that verifies dataset structure and loads the dataset eagerly if lazyload is False.
    """

    if not self.load_images and not self.load_labels:
      raise ValueError("DatasetLoader must load either images or labels or both")

    self.verify_dataset_structure(self.root_path)

    # would be better to initialize the dataset when g
    # if self._validated and not self.lazy_load:
    #   self._dataset = self._load_dataset_eager(self.root_path)

    self.initialized = True

  def set_attributes(self, **kwargs: sys.Any) -> None:
    """
    Sets multiple attributes of the DatasetLoader at once.

    Args:
        **kwargs: A dictionary of attribute names and values.
    """
    current_attributes = [attr for attr in dir(self) if not callable(getattr(self, attr)) and not attr.startswith("__")]

    for key, value in kwargs.items():
      if not key in current_attributes:
        raise AttributeError(f"Attribute {key} not found in DatasetLoader")

      setattr(self, key, value)

  # def same thing as setAttr but using __call__

  def __call__(self, *args: sys.Any, **kwds: sys.Any) -> sys.Any:
    pass

  def update_path(self, path: str) -> None:
    """
    Updates the dataset path and re-validates and loads the dataset if needed.

    Args:
        path (str): The new dataset path.
    """
    self.root_path = path
    self.verify_dataset_structure(self.root_path)
    if self._validated:
      self._dataset = self.loadDataset(self.root_path)

  # [ ] add validation for lable path when load_labels is True
  def verify_dataset_structure(self, path: str) -> None:
    """
    Verifies that the dataset's structure matches the expected format.

    If the dataset has already been validated and the path hasn't changed, 
    the function will return early.

    Args:
        path (str): The root path of the dataset to verify.
    """
    if self._validated and self.root_path == path:
      return

    for folder in os.listdir(path):
      subset_path = os.path.join(path, folder)

      if not os.path.isdir(subset_path):
        continue  # Ignore non-directory files

      for category_folder in os.listdir(subset_path):
        category_path = os.path.join(subset_path, category_folder)
        images_path = os.path.join(category_path, 'images')

        if category_folder in DirectoryConstraints['ignored']:
          continue  # Skip ignored files/folders like .DS_Store

        if not os.path.isdir(category_path):
          logging.warning(f"Found non-directory file: {category_path}")
          logging.warning(f"Subsets must contain category folders {DirectoryConstraints['expected']}")

        if category_folder not in DirectoryConstraints['expected']:
          logging.warning(f"Found unexpected category folder: {category_path}")
          logging.warning(f"Expected categories: {DirectoryConstraints['expected']}")

        if not os.path.exists(images_path):
          logging.warning(f"Images directory not found")
          logging.warning(f"Expected images directory at: {images_path}")

        if not DatasetLoader.allExtensionsUniform(images_path):
          logging.warning(f"Found non-uniform file extensions in category folder: {category_path}")
          if self.skip_non_uniform_directory:
            logging.warning(f"Skipping non-uniform directory")
            continue
          self._validated = False
          return

    logging.info(f"Dataset structure validation complete: {self.root_path}")
    self._validated = True

  def get_iterator(self) -> Iterator[DatasetEntry]:
    """
    Returns an iterator over the dataset entries, either lazy or eager-loaded.

    Raises:
        Exception: If dataset structure is not validated.

    Returns:
        Iterator[DatasetEntry]: An iterator over DatasetEntry objects.
    """
    if not self.skip_validation and not self._validated:
      raise Exception("Dataset structure not validated, cannot return iterator")

    if self.lazy_load:
      return self._load_dataset_lazy(self.root_path)  # Lazy load: yields entries one by one
    return iter(self._dataset)  # Eager load: return pre-loaded dataset

  def _load_dataset_core(self, path: str) -> Iterator[DatasetEntry]:
    """
    Core dataset loading function that traverses the dataset directory 
    and yields DatasetEntry objects.

    Args:
        path (str): The root path of the dataset.

    Returns:
        Iterator[DatasetEntry]: Yields DatasetEntry objects representing images and labels.
    """
    if not self.skip_validation and not self._validated:
      logging.error(f"Dataset structure not validated, cannot load dataset: {path}")
      return

    if self.structured:
      # Recursively find all image files in the dataset path and create DatasetEntry objects
      for image_file in Path(path).rglob('*'):
        if image_file.is_file() and image_file.suffix in ['.jpg', '.jpeg', '.png']:

          image_path = image_file.relative_to(path)
          image_data = cv2.imread(str(image_file)) if self.load_images else None

          # label_target_path = image_file.parent.parent / 'labels' / f'{image_path.stem}.json'
          label_target_path = image_path.parent.parent / 'labels' / image_path.with_suffix('.json').name
          label_path = label_target_path if self.load_lables and label_target_path.exists() else None

          if self.load_label_data and not label_path:
            if self.skip_missing_labels:
              logging.warning(f"Label file not found: {label_target_path}, skipping entry")
              continue
            raise FileNotFoundError(f"Label file not found: {label_target_path}")

          label_data = json.load(label_target_path.open('r')) if self.load_label_data and label_path else None

          category = image_file.parts[-3]  # Assumes directory structure: subset/category/images/file
          subset = image_file.parts[-4]

          # yield DatasetEntry(image_file, label_destination, category, subset)
          yield DatasetEntry(
              image_path=image_path,
              image_data=image_data,
              label_target_path=label_target_path,
              label_path=label_path,
              label_data=label_data,
              category=category,
              subset=subset
          )

  def _load_dataset_eager(self, path: str, batch_size=1000) -> list[DatasetEntry]:
    """
    Eagerly loads the entire dataset into memory. Associated with lazy_load=False.

    Args:
        path (str): The dataset root path.
        batch_size (int, optional): The size of each batch to load (not currently used).

    Returns:
        list[DatasetEntry]: A list of all dataset entries.
    """
    dataset = []
    for entry in self._load_dataset_core(path):
      dataset.append(entry)

    logging.info(f"Loaded {len(dataset)} entries from dataset: {path}")
    return dataset

  def _load_dataset_lazy(self, path: str) -> Iterator[DatasetEntry]:
    """
    Lazily loads the dataset, yielding entries one at a time.

    Args:
        path (str): The dataset root path.

    Returns:
        Iterator[DatasetEntry]: An iterator that yields DatasetEntry objects lazily.
    """
    for entry in self._load_dataset_core(path):
      logging.debug(f"Lazy loaded: {entry.image_path.relative_to(path)}")
      yield entry

  @staticmethod
  def allExtensionsUniform(path: str) -> bool:
    """
    Checks if all file extensions in a directory are the same.

    Args:
        path (str): The path to the directory containing files.

    Returns:
        bool: True if all files in the directory have the same extension, otherwise False.
    """
    extension_frequencies = {}

    for file in os.listdir(path):
      file_path = os.path.join(path, file)
      file_extension = os.path.splitext(file)[1]

      if not os.path.isfile(file_path):
        logging.debug(f"Not a file: {file_path}")
        return False

      if not file_extension:
        logging.debug(f"No extension found for file: {file_path}")
        return False

      if not extension_frequencies:
        extension_frequencies[file_extension] = 1
      elif file_extension in extension_frequencies:
        extension_frequencies[file_extension] += 1
      else:
        logging.debug(f"Foreign extension found: {file_path}, Expected: {extension_frequencies.keys()}")
        return False

    return True

  def __iter__(self):
    """
    Provides an iterator interface to the DatasetLoader, 
    enabling iteration over DatasetEntry objects.
    """
    return self.get_iterator()


if __name__ == "__main__":
  # logging
  logging.basicConfig(level=logging.DEBUG)
  path = sys.argv[1]

  gestureDataset = DatasetLoader(path)
  for entry in gestureDataset:
    print(entry.image_path)
    print(entry.label_target_path)
    print(entry.category)
    print(entry.subset)
    print()
