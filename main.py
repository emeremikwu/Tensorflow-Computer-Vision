import logging
import sys
from dataset_tools.loader import DatasetLoader
from dataset_tools.labeler import DatasetLabeler
from unittests.dataset_generator import generate_temporary_dataset

if __name__ == "__main__":
  logging.basicConfig(level=logging.DEBUG)
  root_path, subsets, cleanup_callback = generate_temporary_dataset(sys.argv[1])

  loader = DatasetLoader(root_path, load_labels=False, load_label_data=False)
  labeler = DatasetLabeler(loader)
  labeler.process_images()
  cleanup_callback()
