import os
import sys
from pathlib import Path, PurePath


if __name__ == "__main__":
    path = Path(sys.argv[2])
    print(path.absolute(), '\n', path.resolve())
    path = path / '1' / 'paper' / 'images' / 'Paper_1.jpg'
    print(path)
    print(path.parent.parent)
    print(path.parts[-3])
    print(path.parts[-4])
    # label_destination = image_file.parent.parent / 'labels'
