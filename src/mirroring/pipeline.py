from src.config import Config
from src.pipeline import Pipeline


class MirroringPipeline(Pipeline):

    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg

    def run(self) -> None:
        raise NotImplementedError()
