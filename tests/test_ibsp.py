import numpy as np
import pytest

from ibsplib.constants import CHAR_ENCODING
from ibsplib.ibsp import *

GEN_CASES_AMOUNT = 5
RNG = np.random.default_rng()

rand_float_arr = lambda s: RNG.random(size=s, dtype=np.float32)
rand_float = lambda: RNG.random(size=1, dtype=np.float32)[0]
rand_int_arr = lambda s: np.random.randint(0, 2**32-1, size=s, dtype=np.uint32)
rand_int = lambda: np.random.randint(0, 2**32-1, dtype=np.uint32)
rand_byte_arr = lambda s: np.random.randint(0, 2**8-1, size=s, dtype=np.ubyte)
arr_eq = lambda arr1, arr2, dtype=None: np.array_equal(np.array(arr1, dtype=dtype), np.array(arr2, dtype=dtype))


class TestLumps:

    @pytest.mark.parametrize('name,flags,contents', [(f'textures/testmap/testtexture_{i}',
                                                      rand_int(),
                                                      rand_int()) for i in range(GEN_CASES_AMOUNT)])
    def test_texture_lump(self, name, flags, contents):
        texture = Texture()
        texture.name = name.encode(CHAR_ENCODING)
        texture.flags = flags
        texture.contents = contents

        assert texture.name == name
        assert texture.flags == flags
        assert texture.contents == contents

    @pytest.mark.parametrize('normal,dist', [(rand_float_arr(3),
                                              rand_float()) for _ in range(GEN_CASES_AMOUNT)])
    def test_plane_lump(self, normal, dist):
        plane = Plane()
        plane.dist = dist
        plane.normal = normal

        assert arr_eq(plane.normal, normal)
        assert plane.dist == dist

    @pytest.mark.parametrize('plane,children,mins,maxs', [(rand_int(),
                                                           rand_int_arr(2),
                                                           rand_int_arr(3),
                                                           rand_int_arr(3)) for _ in range(GEN_CASES_AMOUNT)])
    def test_node_lump(self, plane, children, mins, maxs):
        node = Node()
        node.plane = plane
        node.children = children
        node.mins = mins
        node.maxs = maxs

        assert node.plane == plane
        assert arr_eq(node.children, children)
        assert arr_eq(node.mins, mins)
        assert arr_eq(node.maxs, maxs)

    @pytest.mark.parametrize('cluster,area,mins,maxs,leafface,n_leaffaces,leafbrush,n_leafbrushes',
                             [(rand_int(),
                               rand_int(),
                               rand_int_arr(3),
                               rand_int_arr(3),
                               rand_int(),
                               rand_int(),
                               rand_int(),
                               rand_int(),) for _ in range(GEN_CASES_AMOUNT)])
    def test_leaf_lump(self, cluster, area, mins, maxs, leafface, n_leaffaces, leafbrush, n_leafbrushes):
        leaf = Leaf()
        leaf.cluster = cluster
        leaf.area = area
        leaf.mins = mins
        leaf.maxs = maxs
        leaf.leafface = leafface
        leaf.n_leaffaces = n_leaffaces
        leaf.leafbrush = leafbrush
        leaf.n_leafbrushes = n_leafbrushes

        assert leaf.cluster == cluster
        assert leaf.area == area
        assert arr_eq(leaf.mins, mins)
        assert arr_eq(leaf.maxs, maxs)
        assert leaf.leafface == leafface
        assert leaf.n_leaffaces == n_leaffaces
        assert leaf.leafbrush == leafbrush
        assert leaf.n_leafbrushes == n_leafbrushes

    @pytest.mark.parametrize('face', [rand_int() for _ in range(GEN_CASES_AMOUNT)])
    def test_leafface_lump(self, face):
        leafface = LeafFace()
        leafface.face = face

        assert leafface.face == face

    @pytest.mark.parametrize('brush', [rand_int() for _ in range(GEN_CASES_AMOUNT)])
    def test_leafbrush_lump(self, brush):
        leafbrush = LeafBrush()
        leafbrush.brush = brush

        assert leafbrush.brush == brush

    @pytest.mark.parametrize('mins,maxs,face,n_faces,brush,n_brushes', [(rand_float_arr(3),
                                                                         rand_float_arr(3),
                                                                         rand_int(),
                                                                         rand_int(),
                                                                         rand_int(),
                                                                         rand_int()) for _ in range(GEN_CASES_AMOUNT)])
    def test_model_lump(self, mins, maxs, face, n_faces, brush, n_brushes):
        model = Model()
        model.mins = mins
        model.maxs = maxs
        model.face = face
        model.n_faces = n_faces
        model.brush = brush
        model.n_brushes = n_brushes

        assert arr_eq(model.mins, mins)
        assert arr_eq(model.maxs, maxs)
        assert model.face == face
        assert model.n_faces == n_faces
        assert model.brush == brush
        assert model.n_brushes == n_brushes

    @pytest.mark.parametrize('brushside,n_brushsides,texture', [(rand_int(),
                                                                 rand_int(),
                                                                 rand_int()) for _ in range(GEN_CASES_AMOUNT)])
    def test_brush_lump(self, brushside, n_brushsides, texture):
        brush = Brush()
        brush.brushside = brushside
        brush.n_brushsides = n_brushsides
        brush.texture = texture

        assert brush.brushside == brushside
        assert brush.n_brushsides == n_brushsides
        assert brush.texture == texture

    @pytest.mark.parametrize('plane,texture', [(rand_int(),
                                                rand_int()) for _ in range(GEN_CASES_AMOUNT)])
    def test_brushside_lump(self, plane, texture):
        brushside = Brushside()
        brushside.plane = plane
        brushside.texture = texture

        assert brushside.plane == plane
        assert brushside.texture == texture

    @pytest.mark.parametrize('position,texcoord,normal,color', [(rand_float_arr(3),
                                                                 rand_float_arr((2, 2)),
                                                                 rand_float_arr(3),
                                                                 rand_byte_arr(4)) for _ in range(GEN_CASES_AMOUNT)])
    def test_vertex_lump(self, position, texcoord, normal, color):
        vertex = Vertex()
        vertex.position = position
        vertex.texcoord = texcoord
        vertex.normal = normal
        vertex.color = color

        assert arr_eq(vertex.position, position)
        assert arr_eq(vertex.texcoord, texcoord)
        assert arr_eq(vertex.normal, normal)
        assert arr_eq(vertex.color, color)

    @pytest.mark.parametrize('offset', [rand_int() for _ in range(GEN_CASES_AMOUNT)])
    def test_maeshvert_lump(self, offset):
        maeshvert = Meshvert()
        maeshvert.offset = offset

        assert maeshvert.offset == offset

    @pytest.mark.parametrize('name,brush,unknown', [(f'test/effect_name_{i}',
                                                     rand_int(),
                                                     rand_int()) for i in range(GEN_CASES_AMOUNT)])
    def test_effect_lump(self, name, brush, unknown):
        effect = Effect()
        effect.name = name.encode(CHAR_ENCODING)
        effect.brush = brush
        effect.unknown = unknown

        assert effect.name == name
        assert effect.brush == brush
        assert effect.unknown == unknown

    @pytest.mark.parametrize('texture,effect,_type,vertex,n_vertexes,meshvert,n_meshverts,'
                             'lm_index,lm_start,lm_size,lm_origin,lm_vecs,normal,size',
                             [(rand_int(),
                               rand_int(),
                               rand_int(),
                               rand_int(),
                               rand_int(),
                               rand_int(),
                               rand_int(),
                               rand_int(),
                               rand_int_arr(2),
                               rand_int_arr(2),
                               rand_float_arr(3),
                               rand_float_arr((2, 3)),
                               rand_float_arr(3),
                               rand_int_arr(2)) for _ in range(GEN_CASES_AMOUNT)])
    def test_face_lump(self, texture, effect, _type, vertex, n_vertexes, meshvert, n_meshverts,
                       lm_index, lm_start, lm_size, lm_origin, lm_vecs, normal, size):
        face = Face()
        face.texture = texture
        face.effect = effect
        face.type = _type
        face.vertex = vertex
        face.n_vertexes = n_vertexes
        face.meshvert = meshvert
        face.n_meshverts = n_meshverts
        face.lm_index = lm_index
        face.lm_start = lm_start
        face.lm_size = lm_size
        face.lm_origin = lm_origin
        face.lm_vecs = lm_vecs
        face.normal = normal
        face.size = size

        assert face.texture == texture
        assert face.effect == effect
        assert face.type == _type
        assert face.vertex == vertex
        assert face.n_vertexes == n_vertexes
        assert face.meshvert == meshvert
        assert face.n_meshverts == n_meshverts
        assert face.lm_index == lm_index
        assert arr_eq(face.lm_start, lm_start)
        assert arr_eq(face.lm_size, lm_size)
        assert arr_eq(face.lm_origin, lm_origin)
        assert arr_eq(face.lm_vecs, lm_vecs)
        assert arr_eq(face.normal, normal)
        assert arr_eq(face.size, size)

    @pytest.mark.parametrize('_map', [rand_byte_arr((128, 128, 3)) for _ in range(GEN_CASES_AMOUNT)])
    def test_lightmap_lump(self, _map):
        lightmap = Lightmap()
        lightmap.map = _map

        assert arr_eq(lightmap.map, _map)

    @pytest.mark.parametrize('ambient,directional,dir', [(rand_byte_arr(3),
                                                          rand_byte_arr(3),
                                                          rand_byte_arr(2)) for _ in range(GEN_CASES_AMOUNT)])
    def test_lightvol_lump(self, ambient, directional, dir):
        lightvol = Lightvol()
        lightvol.ambient = ambient
        lightvol.directional = directional
        lightvol.dir = dir

        assert arr_eq(lightvol.ambient, ambient)
        assert arr_eq(lightvol.directional, directional)
        assert arr_eq(lightvol.dir, dir)


class TestIBSP:

    @pytest.mark.parametrize('offset,length', [(rand_int(),
                                                rand_int()) for _ in range(GEN_CASES_AMOUNT)])
    def test_direntry(self, offset, length):
        direntry = DirEntry()
        direntry.offset = offset
        direntry.length = length

        assert direntry.offset == offset
        assert direntry.length == length

    @pytest.mark.parametrize('magic,version', [('IBSP',
                                                rand_int()) for _ in range(GEN_CASES_AMOUNT)])
    def test_header(self, magic, version):
        header = IBSPHeader()
        header.magic = magic.encode(CHAR_ENCODING)
        header.version = version
        direntries = [DirEntry(rand_int(), rand_int()) for _ in range(17)]
        header.direntry = direntries

        assert header.magic == magic
        assert header.version == version
        assert arr_eq(header.direntry, direntries, type(DirEntry))
