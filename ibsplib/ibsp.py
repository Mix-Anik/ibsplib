"""
Python library for working with Quake 3 IBSP structures
Reference sheet: http://www.mralligator.com/q3/
"""
__all__ = ["Entities", "Texture", "Plane", "Node", "Leaf", "LeafFace", "LeafBrush", "Model", "Brush", "Brushside", "Vertex", "Meshvert",
           "Effect", "Face", "Lightmap", "Lightvol", "Visdata", "DirEntry", "IBSPHeader", "IBSP"]

from ctypes import c_byte, c_uint32, c_float, c_char
from pathlib import Path
from typing import List, cast, Any

from ibsplib.constants import *
from ibsplib.base import *


# %% [Lump class definitions]

class Entities(EntitiesABC):
    """
    The entities lump stores game-related map information, including information about
    the map name, weapons, health, armor, triggers, spawn points, lights, and .md3 models to be placed in the map.
    The lump contains only one record, a string that describes all the entities

    Attributes
    ----------
    ents : c_char[entities lump size]
        Entity descriptions, stored as a string
    """

    def __init__(self, data: bytearray):
        super().__init__()
        self._ents = (c_char * len(data)).from_buffer(data)

    @EntitiesABC.ents.getter
    def ents(self) -> str:
        return self._ents.value.decode('ascii')

    @property
    def sz(self) -> int:
        return len(self._ents)


class Texture(TextureABC):
    """
    The textures lump stores information about surfaces and volumes, which are in turn associated with faces, brushes, and brushsides.
    There are a total of lump_size / sizeof(texture) records in the lump.
    """

    _fields_ = [
        ('_name', c_char * 64),  # Texture name (64 bytes)
        ('flags', c_uint32),  # Surface flags
        ('contents', c_uint32)  # Content flags
    ]
    sz = 72

    @TextureABC.name.getter
    def name(self) -> str:
        return self._name.decode('ascii')


class Plane(PlaneABC):
    """
    The planes lump stores a generic set of planes that are in turn referenced by nodes and brushsides.
    There are a total of lump_size / sizeof(plane) records in the lump.
    """

    _fields_ = [
        ('normal', c_float * 3),  # Plane normal
        ('dist', c_float)  # Distance from origin to plane along normal
    ]
    sz = 16


class Node(NodeABC):
    """
    The nodes lump stores all of the nodes in the map's BSP tree.
    The BSP tree is used primarily as a spatial subdivision scheme, dividing the world into convex regions called leafs.
    The first node in the lump is the tree's root node. There are a total of lump_size / sizeof(node) records in the lump.
    """

    _fields_ = [
        ('plane', c_uint32),  # Plane index
        ('children', c_uint32 * 2),  # Children indices. Negative numbers are leaf indices: -(leaf+1)
        ('mins', c_uint32 * 3),  # Integer bounding box min coord
        ('maxs', c_uint32 * 3)  # Integer bounding box max coord
    ]
    sz = 36


class Leaf(LeafABC):
    """
    The leafs lump stores the leaves of the map's BSP tree. Each leaf is a convex region that contains, among other things,
    a cluster index (for determining the other leafs potentially visible from within the leaf), a list of faces (for rendering),
    and a list of brushes (for collision detection). There are a total of lump_size / sizeof(leaf) records in the lump.
    """

    _fields_ = [
        ('cluster', c_uint32),  # Visdata cluster index
        ('area', c_uint32),  # Areaportal area
        ('mins', c_uint32 * 3),  # Integer bounding box min coord
        ('maxs', c_uint32 * 3),  # Integer bounding box max coord
        ('leafface', c_uint32),  # First leafface for leaf
        ('n_leaffaces', c_uint32),  # Number of leaffaces for leaf.
        ('leafbrush', c_uint32),  # First leafbrush for leaf
        ('n_leafbrushes', c_uint32)  # Number of leafbrushes for leaf
    ]
    sz = 48


class LeafFace(LeafFaceABC):
    """
    The leaffaces lump stores lists of face indices, with one list per leaf.
    There are a total of lump_size / sizeof(leafface) records in the lump.
    """

    _fields_ = [
        ('face', c_uint32)  # Face index
    ]
    sz = 4


class LeafBrush(LeafBrushABC):
    """
    The leafbrushes lump stores lists of brush indices, with one list per leaf.
    There are a total of lump_size / sizeof(leafbrush) records in the lump.
    """

    _fields_ = [
        ('brush', c_uint32)  # Brush index
    ]
    sz = 4


class Model(ModelABC):
    """
    The models lump describes rigid groups of world geometry. The first model correponds to the base portion of the map
    while the remaining models correspond to movable portions of the map, such as the map's doors, platforms, and buttons.
    Each model has a list of faces and list of brushes; these are especially important for the movable parts of the map,
    which (unlike the base portion of the map) do not have BSP trees associated with them.
    There are a total of lump_size / sizeof(models) records in the lump.
    """

    _fields_ = [
        ('mins', c_float * 3),  # Bounding box min coord
        ('maxs', c_float * 3),  # Bounding box max coord
        ('face', c_uint32),  # First face for model
        ('n_faces', c_uint32),  # Number of faces for model
        ('brush', c_uint32),  # First brush for model
        ('n_brushes', c_uint32)  # Number of brushes for model
    ]
    sz = 40


class Brush(BrushABC):
    """
    The brushes lump stores a set of brushes, which are in turn used for collision detection.
    Each brush describes a convex volume as defined by its surrounding surfaces.
    There are a total of lump_size / sizeof(brushes) records in the lump.
    """

    _fields_ = [
        ('brushside', c_uint32),  # First brushside for brush
        ('n_brushsides', c_uint32),  # Number of brushsides for brush
        ('texture', c_uint32)  # Texture index
    ]
    sz = 12


class Brushside(BrushsideABC):
    """
    The brushsides lump stores descriptions of brush bounding surfaces.
    There are a total of lump_size / sizeof(brushsides) records in the lump.
    """

    _fields_ = [
        ('plane', c_uint32),  # Plane index
        ('texture', c_uint32)  # Texture index
    ]
    sz = 8


class Vertex(VertexABC):
    """
    The vertexes lump stores lists of vertices used to describe faces.
    There are a total of lump_size / sizeof(vertex) records in the lump.
    """

    _fields_ = [
        ('position', c_float * 3),  # Vertex position
        ('texcoord', (c_float * 2) * 2),  # Vertex texture coordinates. 0=surface, 1=lightmap
        ('normal', c_float * 3),  # Vertex normal
        ('color', c_byte * 4)  # Vertex color. RGBA
    ]
    sz = 44


class Meshvert(MeshvertABC):
    """
    The meshverts lump stores lists of vertex offsets, used to describe generalized triangle meshes.
    There are a total of lump_size / sizeof(meshvert) records in the lump.
    """

    _fields_ = [
        ('offset', c_uint32)  # Vertex index offset, relative to first vertex of corresponding face
    ]
    sz = 4


class Effect(EffectABC):
    """
    The effects lump stores references to volumetric shaders (typically fog) which affect the rendering of a particular group of faces.
    There are a total of lump_size / sizeof(effect) records in the lump.
    """

    _fields_ = [
        ('_name', c_char * 64),  # Effect shader (64 bytes)
        ('brush', c_uint32),  # Brush that generated this effect
        ('unknown', c_uint32)  # Always 5, except in q3dm8, which has one effect with -1
    ]
    sz = 72

    @EffectABC.name.getter
    def name(self) -> str:
        return self._name.decode('ascii')


class Face(FaceABC):
    """
    The faces lump stores information used to render the surfaces of the map.
    There are a total of lump_size / sizeof(faces) records in the lump.
    """

    _fields_ = [
        ('texture', c_uint32),  # Texture index
        ('effect', c_uint32),  # Index into lump 12 (Effects), or -1
        ('type', c_uint32),  # Face type. 1=polygon, 2=patch, 3=mesh, 4=billboard
        ('vertex', c_uint32),  # Index of first vertex
        ('n_vertexes', c_uint32),  # Number of vertices
        ('meshvert', c_uint32),  # Index of first meshvert
        ('n_meshverts', c_uint32),  # Number of meshverts
        ('lm_index', c_uint32),  # Lightmap index
        ('lm_start', c_uint32 * 2),  # Corner of this face's lightmap image in lightmap
        ('lm_size', c_uint32 * 2),  # Size of this face's lightmap image in lightmap
        ('lm_origin', c_float * 3),  # World space origin of lightmap
        ('lm_vecs', (c_float * 3) * 2),  # World space lightmap s and t unit vectors
        ('normal', c_float * 3),  # Surface normal
        ('size', c_uint32 * 2)  # Patch dimensions
    ]
    sz = 104


class Lightmap(LightmapABC):
    """
    The lightmaps lump stores the light map textures used make surface lighting look more realistic.
    There are a total of lump_size / sizeof(lightmap) records in the lump.
    """

    _fields_ = [
        ('map', ((c_byte * 3) * 128) * 128)  # Lightmap color data. RGB
    ]
    sz = 49152


class Lightvol(LightvolABC):
    """
    The lightvols lump stores a uniform grid of lighting information used to illuminate non-map objects.
    There are a total of lump_size / sizeof(lightvol) records in the lump.

    Lightvols make up a 3D grid whose dimensions are:
        nx = floor(models[0].maxs[0] / 64) - ceil(models[0].mins[0] / 64) + 1
        ny = floor(models[0].maxs[1] / 64) - ceil(models[0].mins[1] / 64) + 1
        nz = floor(models[0].maxs[2] / 128) - ceil(models[0].mins[2] / 128) + 1
    """

    _fields_ = [
        ('ambient', c_byte * 3),  # Ambient color component. RGB
        ('directional', c_byte * 3),  # Directional color component. RGB
        ('dir', c_byte * 2),  # Direction to light. 0=phi, 1=theta
    ]
    sz = 8


class Visdata(VisdataABC):
    """
    The visdata lump stores bit vectors that provide cluster-to-cluster visibility information.
    There is exactly one visdata record, with a length equal to that specified in the lump directory.

    Attributes
    ----------
    n_vecs : c_uint32
        Number of vectors
    sz_vecs : c_uint32
        Size of each vector, in bytes
    vecs : c_byte[n_vecs * sz_vecs]
        Visibility data. One bit per cluster per vector
    """

    def __init__(self, data: bytearray):
        super().__init__()
        n_vecs_data = data[:4]
        sz_vecs_data = data[4:8]
        vecs_data = data[8:]
        self._n_vecs = c_uint32.from_buffer(n_vecs_data)
        self._sz_vecs = c_uint32.from_buffer(sz_vecs_data)
        total_vecs_size = self._n_vecs.value * self._sz_vecs.value
        self._vecs = (c_byte * total_vecs_size).from_buffer(vecs_data)

    @VisdataABC.n_vecs.getter
    def n_vecs(self) -> int:
        return self._n_vecs.value

    @VisdataABC.sz_vecs.getter
    def sz_vecs(self) -> int:
        return self._sz_vecs.value

    @VisdataABC.vecs.getter
    def vecs(self) -> List[int]:
        return cast(List[int], self._vecs)

    @property
    def sz(self) -> int:
        return self._n_vecs.value * self._sz_vecs.value


# %% [Main class definitions]

class DirEntry(DirEntryABC):
    """ Each direntry locates a single lump in the BSP file """

    _fields_ = [
        ('offset', c_uint32),  # Offset to start of lump, relative to beginning of file
        ('length', c_uint32)  # Length of lump. Always a multiple of 4
    ]
    sz = 8


class IBSPHeader(IBSPHeaderABC):
    """ IBSP header class """

    _fields_ = [
        ('_magic', c_char * 4),  # Magic number. Always "IBSP"
        ('version', c_uint32),  # Version number. 0x2e for the BSP files distributed with Quake 3
        ('direntry', DirEntry * 17)  # Lump directory, seventeen entries
    ]
    sz = 144

    @IBSPHeaderABC.magic.getter
    def magic(self) -> str:
        return self._magic.decode('ascii')


class IBSP(IBSPABC):
    """
    IBSP structure base class

    Attributes
    ----------
    header : IBSPHeader
        BSP header instance
    data : bytearray
        BSP file byte data

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

    def __init__(self, data: bytearray):
        super().__init__()
        self._header = IBSPHeader.from_buffer(data[:IBSPHeader.sz])
        self._data = data
        self._lump_cache = {}

    @property
    def header(self) -> IBSPHeader:
        return self._header

    @property
    def data(self) -> bytes:
        return self._data

    @property
    def sz(self) -> int:
        return len(self._data)

    def save(self, filepath: str) -> None:
        """ Saves current bsp to file """

        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, 'wb') as f:
            f.write(self._data)

    def get_lump_entries(self, lump_id: int) -> List[Any]:
        """ Returns list of requested lump entries """

        if not 0 <= lump_id <= 16:
            raise ValueError(f'Lump with index "{lump_id}" does not exist')

        if lump_id in self._lump_cache:
            return self._lump_cache[lump_id]

        direntry = self._header.direntry[lump_id]
        lump_entry_cls = self.lump_cls_map[lump_id]
        raw_lump_data = bytearray(self._data[direntry.offset: direntry.offset + direntry.length])
        lump_entries = []

        if lump_id in [ENTITIES_LUMP, VISDATA_LUMP]:
            lump_entries.append(lump_entry_cls(raw_lump_data))
        else:
            entry_size = cast(int, lump_entry_cls.sz)
            total_lump_entries = int(direntry.length / entry_size)

            for i in range(total_lump_entries):
                raw_entry_data = raw_lump_data[entry_size * i: entry_size * (i + 1)]
                lump_entries.append(lump_entry_cls.from_buffer(raw_entry_data))

        # cache result for reuse
        self._lump_cache[lump_id] = lump_entries

        return lump_entries

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
