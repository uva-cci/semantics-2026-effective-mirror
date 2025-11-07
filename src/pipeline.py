from typing import Protocol


class Pipeline(Protocol):
    def run(self) -> None: ...
