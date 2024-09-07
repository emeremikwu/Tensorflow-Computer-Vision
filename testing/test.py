import os
import sys
from pathlib import Path, PurePath


class iterTest:
    def __init__(self, data):
        self.data = data
        self.index = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.index >= len(self.data):
            raise StopIteration
        result = self.data[self.index]
        self.index += 1
        return result


if __name__ == "__main__":
    data = [1, 2, 3, 4, 5]
    # test = iterTest(data)
    # test2 = iter(test)

    # print(next(test))
    # print(next(test2))
    # print(next(test2))

    t1 = iter(data)
    t2 = iter(data)

    print(next(t1))
    print(next(t2))
    print(next(t2))
    print(next(t2))
    print(next(t1))
