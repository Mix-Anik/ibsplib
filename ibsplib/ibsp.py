"""
Python library for working with Quake 3 IBSP structures
Reference sheet: http://www.mralligator.com/q3/
"""
__all__ = ["Entities", "Texture", "Plane", "Node", "Leaf", "LeafFace", "LeafBrush", "Model", "Brush", "Brushside", "Vertex", "Meshvert",
           "Effect", "Face", "Lightmap", "Lightvol", "Visdata", "DirEntry", "IBSPHeader", "IBSP"]

import os

from ctypes import c_byte, c_uint32, c_float, c_char
from pathlib import Path
from typing import List, cast, Any

import numpy as np
import numpy.typing as npt

from ibsplib.constants import *
from ibsplib.base import *
from ibsplib.exceptions import LumpIndexOutOfBounds, ErrorReadingBSPFile, FileDoesNotExist, NotIBSPFileException, UnsupportedBSPFileVersion


# %% [Lump class definitions]

class Entities(BaseStructure):
    """
    The entities lump stores game-related map information, including information about
    the map name, weapons, health, armor, triggers, spawn points, lights, and .md3 models to be placed in the map.
    The lump contains only one record, a string that describes all the entities

    Attributes
    ----------
    ents : str
        Entity descriptions, stored as a string
    sz : np.uint32
        Structure's data total size
    """

    def __init__(self, data: bytearray):
        super().__init__()
        self.__ents = (c_char * len(data)).from_buffer(data)

    @property
    def ents(self) -> str:
        return self.__ents.value.decode(CHAR_ENCODING)

    @property
    def sz(self) -> np.uint32:
        return np.uint32(len(self.__ents))


class Texture(BaseStructure):
    """
    The textures lump stores information about surfaces and volumes, which are in turn associated with faces, brushes, and brushsides.
    There are a total of lump_size/Texture.sz records in the lump.

    Attributes
    ----------
    name : str
        Texture name (64 bytes)
    flags : np.uint32
        Surface flags
    contents : np.uint32
        Content flags
    sz : np.uint32
        Structure's data total size
    """

    _fields_ = [
        ('_name', c_char * 64),
        ('_flags', c_uint32),
        ('_contents', c_uint32)
    ]
    _sz = 72

    @property
    def name(self) -> str:
        return self._name.decode(CHAR_ENCODING)

    @property
    def flags(self) -> np.uint32:
        return np.uint32(self._flags)

    @property
    def contents(self) -> np.uint32:
        return np.uint32(self._contents)


class Plane(BaseStructure):
    """
    The planes lump stores a generic set of planes that are in turn referenced by nodes and brushsides.
    There are a total of lump_size/Plane.sz records in the lump.

    Attributes
    ----------
    normal : npt.NDArray[np.single]
        Plane normal
    dist : np.single
        Distance from origin to plane along normal
    sz : np.uint32
        Structure's data total size
    """

    _fields_ = [
        ('_normal', c_float * 3),
        ('_dist', c_float)
    ]
    _sz = 16

    @property
    def normal(self) -> npt.NDArray[np.single]:
        return np.array(self._normal, dtype=np.single)

    @property
    def dist(self) -> np.single:
        return np.single(self._dist)


class Node(BaseStructure):
    """
    The nodes lump stores all of the nodes in the map's BSP tree.
    The BSP tree is used primarily as a spatial subdivision scheme, dividing the world into convex regions called leafs.
    The first node in the lump is the tree's root node. There are a total of lump_size/Node.sz records in the lump.

    Attributes
    ----------
    plane : np.uint32
        Plane index
    children : npt.NDArray[np.uint32]
        Children indices. Negative numbers are leaf indices: -(leaf+1)
    mins : npt.NDArray[np.uint32]
        Integer bounding box min coord
    maxs : npt.NDArray[np.uint32]
        Integer bounding box max coord
    sz : np.uint32
        Structure's data total size
    """

    _fields_ = [
        ('_plane', c_uint32),
        ('_children', c_uint32 * 2),
        ('_mins', c_uint32 * 3),
        ('_maxs', c_uint32 * 3)
    ]
    _sz = 36

    @property
    def plane(self) -> np.uint32:
        return np.uint32(self._plane)

    @property
    def children(self) -> npt.NDArray[np.uint32]:
        return np.array(self._children, dtype=np.uint32)

    @property
    def mins(self) -> npt.NDArray[np.uint32]:
        return np.array(self._mins, dtype=np.uint32)

    @property
    def maxs(self) -> npt.NDArray[np.uint32]:
        return np.array(self._maxs, dtype=np.uint32)


class Leaf(BaseStructure):
    """
    The leafs lump stores the leaves of the map's BSP tree. Each leaf is a convex region that contains, among other things,
    a cluster index (for determining the other leafs potentially visible from within the leaf), a list of faces (for rendering),
    and a list of brushes (for collision detection). There are a total of lump_size/Leaf.sz records in the lump.

    Attributes
    ----------
    cluster : np.uint32
        Visdata cluster index
    area : np.uint32
        Areaportal area
    mins : npt.NDArray[np.uint32]
        Integer bounding box min coord
    maxs : npt.NDArray[np.uint32]
        Integer bounding box max coord
    leafface : np.uint32
        First leafface for leaf
    n_leaffaces : np.uint32
        Number of leaffaces for leaf
    leafbrush : np.uint32
        First leafbrush for leaf
    n_leafbrushes : np.uint32
        Number of leafbrushes for leaf
    sz : np.uint32
        Structure's data total size
    """

    _fields_ = [
        ('_cluster', c_uint32),
        ('_area', c_uint32),
        ('_mins', c_uint32 * 3),
        ('_maxs', c_uint32 * 3),
        ('_leafface', c_uint32),
        ('_n_leaffaces', c_uint32),
        ('_leafbrush', c_uint32),
        ('_n_leafbrushes', c_uint32)
    ]
    _sz = 48

    @property
    def cluster(self) -> np.uint32:
        return np.uint32(self._cluster)

    @property
    def area(self) -> np.uint32:
        return np.uint32(self._area)

    @property
    def mins(self) -> npt.NDArray[np.uint32]:
        return np.array(self._mins, dtype=np.uint32)

    @property
    def maxs(self) -> npt.NDArray[np.uint32]:
        return np.array(self._maxs, dtype=np.uint32)

    @property
    def leafface(self) -> np.uint32:
        return np.uint32(self._leafface)

    @property
    def n_leaffaces(self) -> np.uint32:
        return np.uint32(self._n_leaffaces)

    @property
    def leafbrush(self) -> np.uint32:
        return np.uint32(self._leafbrush)

    @property
    def n_leafbrushes(self) -> np.uint32:
        return np.uint32(self._n_leafbrushes)


class LeafFace(BaseStructure):
    """
    The leaffaces lump stores lists of face indices, with one list per leaf.
    There are a total of lump_size/LeafFace.sz records in the lump.

    Attributes
    ----------
    face : np.uint32
        Face index
    sz : np.uint32
        Structure's data total size
    """

    _fields_ = [
        ('_face', c_uint32)
    ]
    _sz = 4

    @property
    def face(self) -> np.uint32:
        return np.uint32(self._face)


class LeafBrush(BaseStructure):
    """
    The leafbrushes lump stores lists of brush indices, with one list per leaf.
    There are a total of lump_size/LeafBrush.sz records in the lump.

    Attributes
    ----------
    brush : np.uint32
        Brush index
    sz : np.uint32
        Structure's data total size
    """

    _fields_ = [
        ('_brush', c_uint32)
    ]
    _sz = 4

    @property
    def brush(self) -> np.uint32:
        return np.uint32(self._brush)


class Model(BaseStructure):
    """
    The models lump describes rigid groups of world geometry. The first model correponds to the base portion of the map
    while the remaining models correspond to movable portions of the map, such as the map's doors, platforms, and buttons.
    Each model has a list of faces and list of brushes; these are especially important for the movable parts of the map,
    which (unlike the base portion of the map) do not have BSP trees associated with them.
    There are a total of lump_size/Model.sz records in the lump.

    Attributes
    ----------
    mins : npt.NDArray[np.single]
        Bounding box min coord
    maxs : npt.NDArray[np.single]
        Bounding box max coord
    face : np.uint32
        First face for model
    n_faces : np.uint32
        Number of faces for model
    brush : np.uint32
        First brush for model
    n_brushes : np.uint32
        Number of brushes for model
    sz : np.uint32
        Structure's data total size
    """

    _fields_ = [
        ('_mins', c_float * 3),
        ('_maxs', c_float * 3),
        ('_face', c_uint32),
        ('_n_faces', c_uint32),
        ('_brush', c_uint32),
        ('_n_brushes', c_uint32)
    ]
    _sz = 40

    @property
    def mins(self) -> npt.NDArray[np.single]:
        return np.array(self._mins, dtype=np.single)

    @property
    def maxs(self) -> npt.NDArray[np.single]:
        return np.array(self._maxs, dtype=np.single)

    @property
    def face(self) -> np.uint32:
        return np.uint32(self._face)

    @property
    def n_faces(self) -> np.uint32:
        return np.uint32(self._n_faces)

    @property
    def brush(self) -> np.uint32:
        return np.uint32(self._brush)

    @property
    def n_brushes(self) -> np.uint32:
        return np.uint32(self._n_brushes)


class Brush(BaseStructure):
    """
    The brushes lump stores a set of brushes, which are in turn used for collision detection.
    Each brush describes a convex volume as defined by its surrounding surfaces.
    There are a total of lump_size/Brush.sz records in the lump.

    Attributes
    ----------
    brushside : np.uint32
        First brushside for brush
    n_brushsides : np.uint32
        Number of brushsides for brush
    texture : np.uint32
        Texture index
    sz : np.uint32
        Structure's data total size
    """

    _fields_ = [
        ('_brushside', c_uint32),
        ('_n_brushsides', c_uint32),
        ('_texture', c_uint32)
    ]
    _sz = 12

    @property
    def brushside(self) -> np.uint32:
        return np.uint32(self._brushside)

    @property
    def n_brushsides(self) -> np.uint32:
        return np.uint32(self._n_brushsides)

    @property
    def texture(self) -> np.uint32:
        return np.uint32(self._texture)


class Brushside(BaseStructure):
    """
    The brushsides lump stores descriptions of brush bounding surfaces.
    There are a total of lump_size/Brushside.sz records in the lump.

    Attributes
    ----------
    plane : np.uint32
        Plane index
    texture : np.uint32
        Texture index
    sz : np.uint32
        Structure's data total size
    """

    _fields_ = [
        ('_plane', c_uint32),
        ('_texture', c_uint32)
    ]
    _sz = 8

    @property
    def plane(self) -> np.uint32:
        return np.uint32(self._plane)

    @property
    def texture(self) -> np.uint32:
        return np.uint32(self._texture)


class Vertex(BaseStructure):
    """
    The vertexes lump stores lists of vertices used to describe faces.
    There are a total of lump_size/Vertex.sz records in the lump.

    Attributes
    ----------
    position : npt.NDArray[np.single]
        Vertex position
    texcoord : npt.NDArray[np.single]
        Vertex texture coordinates. 0=surface, 1=lightmap
    normal : npt.NDArray[np.single]
        Vertex normal
    color : npt.NDArray[np.byte]
        Vertex color. RGBA
    sz : np.uint32
        Structure's data total size
    """

    _fields_ = [
        ('_position', c_float * 3),
        ('_texcoord', (c_float * 2) * 2),
        ('_normal', c_float * 3),
        ('_color', c_byte * 4)
    ]
    _sz = 44

    @property
    def position(self) -> npt.NDArray[np.single]:
        return np.array(self._position, dtype=np.single)

    @property
    def texcoord(self) -> npt.NDArray[np.single]:
        return np.array(self._texcoord, dtype=np.single)

    @property
    def normal(self) -> npt.NDArray[np.single]:
        return np.array(self._normal, dtype=np.single)

    @property
    def color(self) -> npt.NDArray[np.byte]:
        return np.array(self._color, dtype=np.byte)


class Meshvert(BaseStructure):
    """
    The meshverts lump stores lists of vertex offsets, used to describe generalized triangle meshes.
    There are a total of lump_size/Meshvert.sz records in the lump.

    Attributes
    ----------
    offset : np.uint32
        Vertex index offset, relative to first vertex of corresponding face
    sz : np.uint32
        Structure's data total size
    """

    _fields_ = [
        ('_offset', c_uint32)
    ]
    _sz = 4

    @property
    def offset(self) -> np.uint32:
        return np.uint32(self._offset)


class Effect(BaseStructure):
    """
    The effects lump stores references to volumetric shaders (typically fog) which affect the rendering of a particular group of faces.
    There are a total of lump_size/Effect.sz records in the lump.

    Attributes
    ----------
    name : np.uint32
        Effect shader (64 bytes)
    brush : np.uint32
        Brush that generated this effect
    unknown : np.uint32
        Always 5, except in q3dm8, which has one effect with -1
    sz : np.uint32
        Structure's data total size
    """

    _fields_ = [
        ('_name', c_char * 64),
        ('_brush', c_uint32),
        ('_unknown', c_uint32)
    ]
    _sz = 72

    @property
    def name(self) -> str:
        return self._name.decode(CHAR_ENCODING)

    @property
    def brush(self) -> np.uint32:
        return np.uint32(self._brush)

    @property
    def unknown(self) -> np.uint32:
        return np.uint32(self._unknown)


class Face(BaseStructure):
    """
    The faces lump stores information used to render the surfaces of the map.
    There are a total of lump_size/Face.sz records in the lump.

    Attributes
    ----------
    texture : np.uint32
        Texture index
    effect : np.uint32
        Index into lump 12 (Effects), or -1
    type : np.uint32
        Face type. 1=polygon, 2=patch, 3=mesh, 4=billboard
    vertex : np.uint32
        Index of first vertex
    n_vertexes : np.uint32
        Number of vertices
    meshvert : np.uint32
        Index of first meshvert
    n_meshverts : np.uint32
        Number of meshverts
    lm_index : np.uint32
        Lightmap index
    lm_start : npt.NDArray[np.uint32]
        Corner of this face's lightmap image in lightmap
    lm_size : npt.NDArray[np.uint32]
        Size of this face's lightmap image in lightmap
    lm_origin : npt.NDArray[np.single]
        World space origin of lightmap
    lm_vecs : npt.NDArray[np.single]
        World space lightmap s and t unit vectors
    normal : npt.NDArray[np.single]
        Surface normal
    size : npt.NDArray[np.uint32]
        Patch dimensions
    sz : np.uint32
        Structure's data total size
    """

    _fields_ = [
        ('_texture', c_uint32),
        ('_effect', c_uint32),
        ('_type', c_uint32),
        ('_vertex', c_uint32),
        ('_n_vertexes', c_uint32),
        ('_meshvert', c_uint32),
        ('_n_meshverts', c_uint32),
        ('_lm_index', c_uint32),
        ('_lm_start', c_uint32 * 2),
        ('_lm_size', c_uint32 * 2),
        ('_lm_origin', c_float * 3),
        ('_lm_vecs', (c_float * 3) * 2),
        ('_normal', c_float * 3),
        ('_size', c_uint32 * 2)
    ]
    _sz = 104

    @property
    def texture(self) -> np.uint32:
        return np.uint32(self._texture)

    @property
    def effect(self) -> np.uint32:
        return np.uint32(self._effect)

    @property
    def type(self) -> np.uint32:
        return np.uint32(self._type)

    @property
    def vertex(self) -> np.uint32:
        return np.uint32(self._vertex)

    @property
    def n_vertexes(self) -> np.uint32:
        return np.uint32(self._n_vertexes)

    @property
    def meshvert(self) -> np.uint32:
        return np.uint32(self._meshvert)

    @property
    def n_meshverts(self) -> np.uint32:
        return np.uint32(self._n_meshverts)

    @property
    def lm_index(self) -> np.uint32:
        return np.uint32(self._lm_index)

    @property
    def lm_start(self) -> npt.NDArray[np.uint32]:
        return np.array(self._lm_start, dtype=np.uint32)

    @property
    def lm_size(self) -> npt.NDArray[np.uint32]:
        return np.array(self._lm_size, dtype=np.uint32)

    @property
    def lm_origin(self) -> npt.NDArray[np.single]:
        return np.array(self._lm_origin, dtype=np.single)

    @property
    def lm_vecs(self) -> npt.NDArray[np.single]:
        return np.array(self._lm_vecs, dtype=np.single)

    @property
    def normal(self) -> npt.NDArray[np.single]:
        return np.array(self._normal, dtype=np.single)

    @property
    def size(self) -> npt.NDArray[np.uint32]:
        return np.array(self._size, dtype=np.uint32)


class Lightmap(BaseStructure):
    """
    The lightmaps lump stores the light map textures used make surface lighting look more realistic.
    There are a total of lump_size/Lightmap.sz records in the lump.

    Attributes
    ----------
    map : npt.NDArray[np.byte]
        Lightmap color data. RGB
    sz : np.uint32
        Structure's data total size
    """

    _fields_ = [
        ('_map', ((c_byte * 3) * 128) * 128)
    ]
    _sz = 49152

    @property
    def map(self) -> npt.NDArray[np.byte]:
        return np.array(self._map, dtype=np.byte)


class Lightvol(BaseStructure):
    """
    The lightvols lump stores a uniform grid of lighting information used to illuminate non-map objects.
    There are a total of lump_size/Lightvol.sz records in the lump.

    Lightvols make up a 3D grid whose dimensions are:
        nx = floor(models[0].maxs[0] / 64) - ceil(models[0].mins[0] / 64) + 1
        ny = floor(models[0].maxs[1] / 64) - ceil(models[0].mins[1] / 64) + 1
        nz = floor(models[0].maxs[2] / 128) - ceil(models[0].mins[2] / 128) + 1

    Attributes
    ----------
    ambient : npt.NDArray[np.byte]
        Ambient color component. RGB
    directional : npt.NDArray[np.byte]
        Directional color component. RGB
    dir : npt.NDArray[np.byte]
        Direction to light. 0=phi, 1=theta
    sz : np.uint32
        Structure's data total size
    """

    _fields_ = [
        ('_ambient', c_byte * 3),
        ('_directional', c_byte * 3),
        ('_dir', c_byte * 2),
    ]
    _sz = 8

    @property
    def ambient(self) -> npt.NDArray[np.byte]:
        return np.array(self._ambient, dtype=np.byte)

    @property
    def directional(self) -> npt.NDArray[np.byte]:
        return np.array(self._directional, dtype=np.byte)

    @property
    def dir(self) -> npt.NDArray[np.byte]:
        return np.array(self._dir, dtype=np.byte)


class Visdata(BaseStructure):
    """
    The visdata lump stores bit vectors that provide cluster-to-cluster visibility information.
    There is exactly one visdata record, with a length equal to that specified in the lump directory.

    Attributes
    ----------
    n_vecs : np.uint32
        Number of vectors
    sz_vecs : np.uint32
        Size of each vector, in bytes
    vecs : npt.NDArray[np.byte]
        Visibility data. One bit per cluster per vector
    sz : np.uint32
        Structure's data total size
    """

    def __init__(self, data: bytearray):
        super().__init__()

        self.__n_vecs = c_uint32.from_buffer(data[:4])
        self._sz_vecs = c_uint32.from_buffer(data[4:8])
        total_vecs_size = self.__n_vecs.value * self._sz_vecs.value
        self.__vecs = (c_byte * total_vecs_size).from_buffer(data[8:])

    @property
    def n_vecs(self) -> np.uint32:
        return np.uint32(self.__n_vecs)

    @property
    def sz_vecs(self) -> np.uint32:
        return np.uint32(self._sz_vecs)

    @property
    def vecs(self) -> npt.NDArray[np.byte]:
        return np.array(self.__vecs, dtype=np.byte)

    @property
    def sz(self) -> np.uint32:
        return np.uint32(self.__n_vecs.value * self._sz_vecs.value)


# %% [Main class definitions]

class DirEntry(BaseStructure):
    """
    Each direntry locates a single lump in the BSP file

    Attributes
    ----------
    offset : np.uint32
        Offset to start of lump, relative to beginning of file
    length : np.uint32
        Length of lump. Always a multiple of 4
    sz : np.uint32
        Structure's data total size
    """

    _fields_ = [
        ('_offset', c_uint32),
        ('_length', c_uint32)
    ]
    _sz = 8

    @property
    def offset(self) -> np.uint32:
        return np.uint32(self._offset)

    @property
    def length(self) -> np.uint32:
        return np.uint32(self._length)


class IBSPHeader(BaseStructure):
    """
    IBSP header class

    Attributes
    ----------
    magic : str
        Magic number. Always "IBSP"
    version : np.uint32
        Version number. 0x2e for the BSP files distributed with Quake 3
    direntry : List[DirEntry]
        Lump directory, seventeen entries
    sz : np.uint32
        Structure's data total size
    """

    _fields_ = [
        ('_magic', c_char * 4),
        ('_version', c_uint32),
        ('_direntry', DirEntry * 17)
    ]
    _sz = 144

    @property
    def magic(self) -> str:
        return self._magic.decode(CHAR_ENCODING)

    @property
    def version(self) -> np.uint32:
        return np.uint32(self._version)

    @property
    def direntry(self) -> List[DirEntry]:
        return list(self._direntry)


class IBSP:
    """
    IBSP structure base class

    Attributes
    ----------
    header : IBSPHeader
        BSP header instance
    data : bytearray
        BSP file byte data
    sz : np.uint32
        BSP file data total size

    respective attributes for each lump : respective lump type

    Methods
    -------
    get_lump_entries(lump_id: int):
        Returns list of requested lump entries
    save(self, filepath: str):
        Saves current bsp to file
    """

    lump_cls_map = {
        0: Entities,
        1: Texture,
        2: Plane,
        3: Node,
        4: Leaf,
        5: LeafFace,
        6: LeafBrush,
        7: Model,
        8: Brush,
        9: Brushside,
        10: Vertex,
        11: Meshvert,
        12: Effect,
        13: Face,
        14: Lightmap,
        15: Lightvol,
        16: Visdata
    }

    def __init__(self, filepath: str):
        self.__data = self._read(filepath)
        self.__header = IBSPHeader.from_buffer(self.__data[:IBSPHeader.sz])
        self.__lump_cache = {}

        if self.__header.magic != 'IBSP':
            raise NotIBSPFileException(f'Header magic number mismatch: found "{self.__header.magic}", but should be "IBSP"')

        if self.__header.version not in [46, 47]:
            raise UnsupportedBSPFileVersion(f'Only [46, 47] bsp versions are supported')

    @staticmethod
    def _read(filepath: str) -> bytearray:
        if not os.path.exists(filepath):
            raise FileDoesNotExist(f'Path does not correspond to an actual file - "{filepath}"')

        try:
            with open(filepath, 'rb') as f:
                raw_data = bytearray(f.read())
        except PermissionError as e:
            raise ErrorReadingBSPFile(f'You should have permissions to read that file: {e}')
        except Exception as e:
            raise ErrorReadingBSPFile(f'Failed to read BSP file: {e}')

        return raw_data

    def save(self, filepath: str) -> None:
        """ Saves current bsp to file """

        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, 'wb') as f:
            f.write(self.__data)

    def get_lump_entries(self, lump_id: int) -> List[Any]:
        """ Returns list of requested lump entries """

        if not 0 <= lump_id <= 16:
            raise LumpIndexOutOfBounds(f'Lump with index "{lump_id}" does not exist')

        if lump_id in self.__lump_cache:
            return self.__lump_cache[lump_id]

        direntry = self.__header.direntry[lump_id]
        lump_entry_cls = self.lump_cls_map[lump_id]
        raw_lump_data = bytearray(self.__data[direntry.offset: direntry.offset + direntry.length])
        lump_entries = []

        if lump_id in [ENTITIES_LUMP, VISDATA_LUMP]:
            lump_entries.append(lump_entry_cls(raw_lump_data))
        else:
            entry_size = cast(np.uint32, lump_entry_cls.sz)
            total_lump_entries = int(direntry.length / entry_size)

            for i in range(total_lump_entries):
                raw_entry_data = raw_lump_data[entry_size * i: entry_size * (i + 1)]
                lump_entries.append(lump_entry_cls.from_buffer(raw_entry_data))

        # cache result for reuse
        self.__lump_cache[lump_id] = lump_entries

        return lump_entries

    @property
    def header(self) -> IBSPHeader:
        return self.__header

    @property
    def data(self) -> bytearray:
        return self.__data

    @property
    def sz(self) -> int:
        return len(self.__data)

    @property
    def entities(self) -> Entities:
        return self.get_lump_entries(ENTITIES_LUMP)[0]

    @property
    def textures(self) -> List[Texture]:
        return self.get_lump_entries(TEXTURES_LUMP)

    @property
    def planes(self) -> List[Plane]:
        return self.get_lump_entries(PLANES_LUMP)

    @property
    def nodes(self) -> List[Node]:
        return self.get_lump_entries(NODES_LUMP)

    @property
    def leafs(self) -> List[Leaf]:
        return self.get_lump_entries(LEAFS_LUMP)

    @property
    def leaffaces(self) -> List[LeafFace]:
        return self.get_lump_entries(LEAFFACES_LUMP)

    @property
    def models(self) -> List[Model]:
        return self.get_lump_entries(MODELS_LUMP)

    @property
    def brushes(self) -> List[Brush]:
        return self.get_lump_entries(BRUSHES_LUMP)

    @property
    def brushsides(self) -> List[Brushside]:
        return self.get_lump_entries(BRUSHSIDES_LUMP)

    @property
    def vertexes(self) -> List[Vertex]:
        return self.get_lump_entries(VERTEXES_LUMP)

    @property
    def meshverts(self) -> List[Meshvert]:
        return self.get_lump_entries(MESHVERTS_LUMP)

    @property
    def effects(self) -> List[Effect]:
        return self.get_lump_entries(EFFECTS_LUMP)

    @property
    def faces(self) -> List[Face]:
        return self.get_lump_entries(FACES_LUMP)

    @property
    def lightmaps(self) -> List[Lightmap]:
        return self.get_lump_entries(LIGHTMAPS_LUMP)

    @property
    def lightvols(self) -> List[Lightvol]:
        return self.get_lump_entries(LIGHTVOLS_LUMP)

    @property
    def visdata(self) -> Visdata:
        return self.get_lump_entries(VISDATA_LUMP)[0]
