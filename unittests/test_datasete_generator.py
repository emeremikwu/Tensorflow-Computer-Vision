import pathlib
import random
import string


def create_and_structure_directory():
  dataset_suffix = random.choice(string.ascii_letters, )
  root_path = pathlib.Path(f"tmp_{dataset_suffix}")
