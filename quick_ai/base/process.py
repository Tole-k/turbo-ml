from abc import ABC, ABCMeta, abstractmethod
from typing import Set


class Process(ABC):
    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def avalible_input_formats(self) -> Set:
        pass

    @abstractmethod
    def avalible_output_formats(self) -> Set:
        pass
