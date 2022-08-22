__all__ = ["BaseStructure"]

import numpy as np

from ctypes import Structure

from ibsplib.exceptions import StructureSizeNotDefined


class BaseStructureMeta(type(Structure)):
    _sz = None

    @property
    def sz(cls) -> np.uint32:
        if not cls._sz:
            raise StructureSizeNotDefined('Every structure should have total byte size defined')

        return np.uint32(cls._sz)


class BaseStructure(Structure, metaclass=BaseStructureMeta):
    pass
