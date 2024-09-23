import sys
import pathlib
import unittest

sys.path.append(str(pathlib.Path(__file__).parent.parent))
from dataset_tools import loader, labeler, utils

# test for dataset_tools.labeler


class TestLabeler(unittest.TestCase):
  def setUp(self):
    self.loader = loader.DatasetLoader()
    self.labeler = labeler.Labeler()
    self.labeler.dataset = self.loader

  def test_dataset_property(self):
    self.assertEqual(self.labeler.dataset, self.loader)

  def test_is_ready_property(self):
    self.assertFalse(self.labeler.is_ready)

  def test_process_images(self):
    self.assertIsNone(self.labeler.process_images())

  def test_label(self):
    self.assertIsNone(self.labeler.label([]))

  def test_check_if_ready(self):
    self.assertIsNone(self.labeler._check_if_ready())
