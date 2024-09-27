
import sys
import pathlib
import unittest
import json
from formats.labeling import LabelEntry


# fmt: off
# root = str(pathlib.Path(__file__).parent.parent)
# sys.path.insert(0, root)

# from dataset_tools.utils import results_to_json, write_json_to_file
from dataset_tools import loader, labeler
from unittests.dataset_generator import generate_temporary_dataset
# fmt: on

# test for dataset_tools.labeler


class LabelerTest(unittest.TestCase):
  @classmethod
  def setUpClass(cls):
    temporary_dataset_path, subsets, cleanup_callback = generate_temporary_dataset("unittests/images")
    cls.dataset_path = temporary_dataset_path
    cls.subsets = subsets
    cls.cleanup_callback = cleanup_callback
    print("tmp dataset path: ", cls.dataset_path)

    cls._ds_loader = loader.DatasetLoader(cls.dataset_path, load_labels=False, load_label_data=False)
    cls._ds_labeler = labeler.DatasetLabeler(cls._ds_loader)
    cls._ds_labeler.process_images()

    cls.ds_images = [image for image in cls.dataset_path.rglob("*.jpg")]
    cls.ds_json_files = [json_file for json_file in cls.dataset_path.rglob("*.json")]

    print(len(cls.ds_images), len(cls.ds_json_files))

  def test_dataset_property(self):
    self.assertEqual(self._ds_loader, self._ds_labeler._dataset_loader)

  def test_images_equal_json_files(self):
    self.assertEqual(len(self.ds_images), len(self.ds_json_files))

  def test_multi_hand_landmarks(self):
    for json_file in self.ds_json_files:
      with self.subTest(f"Checking {json_file.name}", json_file=json_file):
        if not "left" in json_file.name or not "right" in json_file.name:
          continue

        with open(json_file, "r") as file:
          data: LabelEntry = json.load(file)
          keypoints = data.get("keypoints", {})

          # check if left and right keypoint exists
          self.assertIn("left", keypoints, f"File {json_file} is missing 'left' keypoints.")
          self.assertIn("right", keypoints, f"File {json_file} is missing 'right' keypoints.")

          # check if left and right keypoint lists are not empty
          self.assertTrue(keypoints["left"], f"File {json_file}: 'left' keypoints should not be an empty list.")
          self.assertTrue(keypoints["right"], f"File {json_file}: 'right' keypoints should not be an empty list.")

  @classmethod
  def tearDownClass(cls):
    cls.cleanup_callback()


if __name__ == '__main__':
  unittest.main()
