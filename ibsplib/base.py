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

    def _cp_arr(self, arr, field_name):
        arr_t = type(arr)

        if not np.issubdtype(arr_t, np.ndarray):
            raise TypeError(f'Expected to get value of type compatible with np.ndarray, but got {arr_t}')

        field_ref = getattr(self, field_name)
        orig_shape = np.shape(field_ref)
        shape = np.shape(arr)

        if shape != orig_shape:
            raise ValueError(f'Given array does not match original shape: expected {orig_shape} but got {shape}')

        for i in range(len(arr)):
            field_ref[i] = arr[i]

    def _cp_num_val(self, val, field_name):
        val_t = type(val)

        if not np.issubdtype(type(val), np.number):
            raise TypeError(f'Expected to get value of type compatible with np.ndarray, but got {val_t}')

        setattr(self, field_name, val)

    def _cp_str_val(self, val, field_name, max_len=None):
        val_t = type(val)

        if not np.issubdtype(type(val), np.character):
            raise TypeError(f'Expected to get value of type compatible with np.character, but got {val_t}')

        if max_len and len(val) > max_len:
            raise ValueError(f'Given string has exceeded max length: {len(val)} > {max_len}')

        setattr(self, field_name, val)
