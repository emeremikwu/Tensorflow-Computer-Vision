import cv2
import logging
import os
import dataclasses
from pathlib import Path
from itertools import islice


# dbg
import sys

DirectoryConstraints = {
    "expected": ['rock', 'paper', 'scissors', 'uncatagorized'],
    "ignored": [".DS_Store"]
}


@dataclasses.dataclass
class DatasetEntry:
    file_path: Path
    label_destinaion_dir: Path = dataclasses.field(default=None)
    category: str = dataclasses.field(default=None)
    subset: str = dataclasses.field(default=None)


@dataclasses.dataclass
class DatasetLoader:

    path: str
    structured = True
    skipValidation = False
    skipNonUniformDirectory = True
    ignoreRandomFiles = True
    lazyload = True

    _validated: bool = dataclasses.field(default=False, init=False)
    _dataset: list = dataclasses.field(default=None, init=False)  # list of DatasetEntry objects, will only be used if lazyload=False

    def __post_init__(self):
        self.verifyDatasetStructure(self.path)
        if self._validated and not self.lazyload:
            self._dataset = self._loadDataset_eager(self.path)

    def updatePath(self, path: str):
        self.path = path
        self.verifyDatasetStructure(self.path)
        if (self._validated):
            self._dataset = self.loadDataset(self.path)

    def verifyDatasetStructure(self, path: str):
        # if the dataset has already been validated and the path has not changed, return
        if self._validated and self.path == path:
            return

        for folder in os.listdir(path):
            subset_path = os.path.join(path, folder)

            if not os.path.isdir(subset_path):
                continue

            # subset path validation
            for category_folder in os.listdir(subset_path):
                category_path = os.path.join(subset_path, category_folder)
                images_path = os.path.join(category_path, 'images')

                if category_folder in DirectoryConstraints['ignored']:
                    # logging.debug(f"Ignoring file/directory: {category_path}")
                    continue

                if not os.path.isdir(category_path):
                    logging.warning(f"Found non-directory file: {category_path}")
                    logging.warning(f"Subsets must be directories containing category folders {DirectoryConstraints['expected']}")

                if category_folder not in DirectoryConstraints['expected']:
                    logging.warning(f"Found unexpected category folder: {category_path}")
                    logging.warning(f"Expected categories: {DirectoryConstraints['expected']}")

                if not os.path.exists(images_path):
                    logging.warning(f"Images directory not found")
                    logging.warning(f"Expected images directory at: {images_path}")

                if not DatasetLoader.allExtensionsUniform(images_path):
                    logging.warning(f"Found non-uniform file extensions in category folder: {category_path}")
                    logging.warning(f"Expected uniform file extensions")
                    if self.skipNonUniformDirectory:
                        logging.warning(f"Skipping non-uniform directory")
                        continue
                    self._validated = False
                    return

            logging.debug(f"Dataset subset structure validation passed: {subset_path}")

        logging.info(f"Dataset structure validation complete: {self.path}")
        self._validated = True

    def getIterator(self):
        if self.lazyload:
            # this solved everything for me, just create a new generator every time 🤯
            return self._loadDataset_lazy(self.path)
        else:
            return self._dataset.__iter__()

    # core dataset loading function, traverses dataset directory and yields dataset entries
    def _loadDataset_traversal(self, path):
        if not self.skipValidation and not self._validated:
            logging.error(f'Dataset structure not validated, cannot load dataset: {path}')
            return None

        if self.structured:
            for subset in [_ for _ in Path(path).iterdir() if _.is_dir()]:
                for category in [_ for _ in subset.iterdir() if _.name in DirectoryConstraints['expected']]:
                    for image_dir in [_ for _ in category.iterdir() if _.is_dir()]:
                        for image_file in [_ for _ in image_dir.iterdir() if _.name not in DirectoryConstraints['ignored']]:
                            label_destination = Path(subset, category, 'labels')
                            yield DatasetEntry(image_file, label_destination, category.name, subset.name)

    # presence of yield causes function to return generator even if lazy=False, must be split into 2 separate operations
    def _loadDataset_eager(self, path):
        dataset = []
        for entry in self._loadDataset_traversal(path):
            dataset.append(entry)

        logging.info(f'Loaded {len(dataset)} entries from dataset: {path}')
        return dataset

    def _loadDataset_lazy(self, path):
        for entry in self._loadDataset_traversal(path):
            logging.debug(f"Lazy loaded: {entry.file_path.relative_to(path)}")
            yield entry

    @staticmethod
    # checks if all file extensions in a directory are the same
    def allExtensionsUniform(path: str) -> bool:

        for file in os.listdir(path):
            file_path = os.path.join(path, file)
            file_extension = os.path.splitext(file)
            extension_frequencies: dict[str, int] = {}

            # check if the file is a file
            if not os.path.isfile(file_path):
                logging.debug(f"not a file: {file_path}")
                return False

            # check if the file has an extension
            if len(file_extension) == 1:
                logging.debug(f"No extension found for file: {file_path}")
                return False

            # check if extension dictionary is empty
            if not len(extension_frequencies):
                extension_frequencies[file_extension] = 1

            # check if the extension is already in the dictionary
            elif file_extension in extension_frequencies:
                extension_frequencies[file_extension] += 1
            else:
                logging.debug(f"Foreign extension found: {file_path}, Expected: {extension_frequencies.keys()}")
                return False

        return True

    def __iter__(self):
        return self.getIterator()


if __name__ == "__main__":
    # logging
    logging.basicConfig(level=logging.DEBUG)
    path = sys.argv[1]

    gestureDataset = DatasetLoader(path)
    for entry in gestureDataset:
        print(entry.file_path)
        print(entry.label_destinaion_dir)
        print(entry.category)
        print(entry.subset)
        print()
