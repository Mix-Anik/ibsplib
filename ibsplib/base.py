__all__ = ["DirEntryABC", "IBSPHeaderABC", "IBSPABC", "EntitiesABC", "TextureABC", "PlaneABC", "NodeABC", "LeafABC", "LeafFaceABC",
           "LeafBrushABC", "ModelABC", "BrushABC", "BrushsideABC","VertexABC", "MeshvertABC", "EffectABC", "FaceABC", "LightmapABC",
           "LightvolABC", "VisdataABC"]

from abc import abstractmethod
from ctypes import Structure
from typing import List


# %% [Type definitions]

class SizeMixin:

    @property
    @abstractmethod
    def sz(self) -> int:
        pass


class DirEntryABC(Structure):

    @property
    @abstractmethod
    def offset(self) -> int:
        pass

    @property
    @abstractmethod
    def length(self) -> int:
        pass


class IBSPHeaderABC(Structure):

    @property
    @abstractmethod
    def magic(self) -> str:
        pass

    @property
    @abstractmethod
    def version(self) -> int:
        pass

    @property
    @abstractmethod
    def direntry(self) -> List[DirEntryABC]:
        pass


class IBSPABC(Structure, SizeMixin):

    @property
    @abstractmethod
    def header(self) -> IBSPHeaderABC:
        pass

    @property
    @abstractmethod
    def data(self) -> bytearray:
        pass


class EntitiesABC(Structure, SizeMixin):

    @property
    @abstractmethod
    def ents(self) -> str:
        pass


class TextureABC(Structure):

    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @property
    @abstractmethod
    def flags(self) -> int:
        pass

    @property
    @abstractmethod
    def contents(self) -> int:
        pass


class PlaneABC(Structure):

    @property
    @abstractmethod
    def normal(self) -> List[float]:
        pass

    @property
    @abstractmethod
    def dist(self) -> float:
        pass


class NodeABC(Structure):

    @property
    @abstractmethod
    def plane(self) -> int:
        pass

    @property
    @abstractmethod
    def children(self) -> List[int]:
        pass

    @property
    @abstractmethod
    def mins(self) -> List[int]:
        pass

    @property
    @abstractmethod
    def maxs(self) -> List[int]:
        pass


class LeafABC(Structure):

    @property
    @abstractmethod
    def cluster(self) -> int:
        pass

    @property
    @abstractmethod
    def area(self) -> int:
        pass

    @property
    @abstractmethod
    def mins(self) -> List[int]:
        pass

    @property
    @abstractmethod
    def maxs(self) -> List[int]:
        pass

    @property
    @abstractmethod
    def leafface(self) -> int:
        pass

    @property
    @abstractmethod
    def n_leaffaces(self) -> int:
        pass

    @property
    @abstractmethod
    def leafbrush(self) -> int:
        pass

    @property
    @abstractmethod
    def n_leafbrushes(self) -> int:
        pass


class LeafFaceABC(Structure):

    @property
    @abstractmethod
    def face(self) -> int:
        pass


class LeafBrushABC(Structure):

    @property
    @abstractmethod
    def brush(self) -> int:
        pass


class ModelABC(Structure):

    @property
    @abstractmethod
    def mins(self) -> List[int]:
        pass

    @property
    @abstractmethod
    def maxs(self) -> List[int]:
        pass

    @property
    @abstractmethod
    def face(self) -> int:
        pass

    @property
    @abstractmethod
    def n_faces(self) -> int:
        pass

    @property
    @abstractmethod
    def brush(self) -> int:
        pass

    @property
    @abstractmethod
    def n_brushes(self) -> int:
        pass


class BrushABC(Structure):

    @property
    @abstractmethod
    def brushside(self) -> int:
        pass

    @property
    @abstractmethod
    def n_brushsides(self) -> int:
        pass

    @property
    @abstractmethod
    def texture(self) -> int:
        pass


class BrushsideABC(Structure):

    @property
    @abstractmethod
    def plane(self) -> int:
        pass

    @property
    @abstractmethod
    def texture(self) -> int:
        pass


class VertexABC(Structure):

    @property
    @abstractmethod
    def position(self) -> List[float]:
        pass

    @property
    @abstractmethod
    def texcoord(self) -> List[List[float]]:
        pass

    @property
    @abstractmethod
    def normal(self) -> List[float]:
        pass

    @property
    @abstractmethod
    def color(self) -> int:
        pass


class MeshvertABC(Structure):

    @property
    @abstractmethod
    def offset(self) -> int:
        pass


class EffectABC(Structure):

    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @property
    @abstractmethod
    def brush(self) -> int:
        pass

    @property
    @abstractmethod
    def unknown(self) -> int:
        pass


class FaceABC(Structure):

    @property
    @abstractmethod
    def texture(self) -> int:
        pass

    @property
    @abstractmethod
    def effect(self) -> int:
        pass

    @property
    @abstractmethod
    def type(self) -> int:
        pass

    @property
    @abstractmethod
    def vertex(self) -> int:
        pass

    @property
    @abstractmethod
    def n_vertexes(self) -> int:
        pass

    @property
    @abstractmethod
    def meshvert(self) -> int:
        pass

    @property
    @abstractmethod
    def n_meshverts(self) -> int:
        pass

    @property
    @abstractmethod
    def lm_index(self) -> int:
        pass

    @property
    @abstractmethod
    def lm_start(self) -> List[int]:
        pass

    @property
    @abstractmethod
    def lm_size(self) -> List[int]:
        pass

    @property
    @abstractmethod
    def lm_origin(self) -> List[float]:
        pass

    @property
    @abstractmethod
    def lm_vecs(self) -> List[List[float]]:
        pass

    @property
    @abstractmethod
    def normal(self) -> List[float]:
        pass

    @property
    @abstractmethod
    def size(self) -> List[int]:
        pass


class LightmapABC(Structure):

    @property
    @abstractmethod
    def map(self) -> List[List[List[int]]]:
        pass


class LightvolABC(Structure):

    @property
    @abstractmethod
    def ambient(self) -> List[int]:
        pass

    @property
    @abstractmethod
    def directional(self) -> List[int]:
        pass

    @property
    @abstractmethod
    def dir(self) -> List[int]:
        pass


class VisdataABC(Structure, SizeMixin):

    @property
    @abstractmethod
    def n_vecs(self) -> int:
        pass

    @property
    @abstractmethod
    def sz_vecs(self) -> int:
        pass

    @property
    @abstractmethod
    def vecs(self) -> List[int]:
        pass
