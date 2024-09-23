from dataclasses import dataclass, field


@dataclass
class A:
  a: list[int] = field(default_factory=lambda: [4, 5, 6])
  b = 20


if __name__ == "__main__":
  a = A(["a", "b", "c"])
  a_vars = [var for var in dir(a) if not var.startswith("__")]
  print(a_vars)
