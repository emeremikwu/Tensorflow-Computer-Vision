from dataclasses import dataclass, field


@dataclass
class A:
    a: list[int] = field(default_factory=lambda: [4, 5, 6])


if __name__ == "__main__":
    a = A(["a", "b", "c"])

    print(a)
