import tempfile
from pathlib import Path
import random
import string
import sys
import logging


def create_tmp_dataset(test_images_path: str) -> tuple[callable, Path, list[Path]]:

  tmp_workspace = tempfile.TemporaryDirectory(prefix="dataset_")
  root_path = Path(tmp_workspace.name)

  logging.debug(f"temporary workspace directory: {tmp_workspace.name}")
  subset_1 = root_path / "1"  # left hand only
  subset_2 = root_path / "subset_2"  # right hand only
  subset_3 = root_path / "".join(random.choices(string.ascii_letters, k=10))  # both hands

  subsets = [subset_1, subset_2, subset_3]

  # doesn't look nice, but it's just for testing purposes
  subset_1_images = [image for image in Path(test_images_path).rglob("*.jpg") if "left" in image.name and "right" not in image.name]
  subset_2_images = [image for image in Path(test_images_path).rglob("*.jpg") if "right" in image.name and "left" not in image.name]
  subset_3_images = [image for image in Path(test_images_path).rglob("*.jpg")]

  for subset in subsets:
    images_path = Path(subset, "paper", "images")
    images_path.mkdir(parents=True, exist_ok=True)

    images: list[Path]

    if subset == subset_1:
      images = subset_1_images
    elif subset == subset_2:
      images = subset_2_images
    elif subset == subset_3:
      images = subset_3_images

    for image in images:
      destination_path = images_path / image.name
      destination_path.write_bytes(image.read_bytes())

  # return cleanup callback function
  return tmp_workspace.cleanup, root_path, subsets


if __name__ == "__main__":
  logging.basicConfig(level=logging.DEBUG)
  callback, _, _ = create_tmp_dataset(sys.argv[1])

  callback()
  print("done")
