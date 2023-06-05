from dataclasses import dataclass

@dataclass
class Linear:
    start: float
    stop: float
    interval: float

    def __iter__(self):
        yield from range(self.start, self.stop, self.interval)





Linear(0, 10, 0.12)



