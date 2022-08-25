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

    def _cp_arr(self, arr, field_name) -> None:
        """ Copies passed n-dim array to structures' field """

        arr_t = type(arr)

        if not np.issubdtype(arr_t, np.ndarray):
            raise TypeError(f'Expected to get value of type compatible with np.ndarray, but got {arr_t}')

        arr = np.array(arr)
        field_ref = getattr(self, field_name)
        orig_shape = np.shape(field_ref)
        shape = np.shape(arr)

        if shape != orig_shape:
            raise ValueError(f'Given array does not match original shape: expected {orig_shape} but got {shape}')

        for idxs, val in np.ndenumerate(arr):
            self.__set_ndarr_value(field_ref, idxs, val)

    def _cp_num_val(self, val, field_name) -> None:
        """ Copies passed numeric value to structures' field """

        val_t = type(val)

        if not np.issubdtype(type(val), np.number):
            raise TypeError(f'Expected to get value of type compatible with np.ndarray, but got {val_t}')

        setattr(self, field_name, val)

    def _cp_str_val(self, val, field_name, max_len=None) -> None:
        """ Copies passed string value to structures' field """

        val_t = type(val)

        if not np.issubdtype(type(val), np.character):
            raise TypeError(f'Expected to get value of type compatible with np.character, but got {val_t}')

        if max_len and len(val) > max_len:
            raise ValueError(f'Given string has exceeded max length: {len(val)} > {max_len}')

        setattr(self, field_name, val)

    def __set_ndarr_value(self, arr, idxs, value) -> None:
        for idx in idxs[:-1]:
            arr = arr[idx]

        arr[idxs[-1]] = value
