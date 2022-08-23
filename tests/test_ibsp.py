import numpy as np
import pytest

from ibsplib import CHAR_ENCODING
from ibsplib.ibsp import *

GEN_CASES_AMOUNT = 5
RNG = np.random.default_rng()


random_vec3 = lambda: RNG.random(size=3, dtype=np.float32)
random_point3 = lambda minv, maxv: np.random.randint(minv, maxv, size=3, dtype=np.uint32)
random_num = lambda minv, maxv: np.random.randint(minv, maxv, dtype=np.uint32)


class TestLumps:
    def setup(self):
        pass

    @pytest.mark.parametrize('name,flags,contents', [(f'textures/testmap/testtexture_{i}',
                                                      random_num(0, 2**31),
                                                      random_num(0, 2**31)) for i in range(GEN_CASES_AMOUNT)])
    def test_texture_lump(self, name, flags, contents):
        texture = Texture()
        texture.name = name.encode(CHAR_ENCODING)
        texture.flags = flags
        texture.contents = contents

        assert texture.name == name
        assert texture.flags == flags
        assert texture.contents == contents

    @pytest.mark.parametrize('normal,dist', [(random_vec3(),
                                              RNG.random(size=1, dtype=np.float32)[0]) for i in range(GEN_CASES_AMOUNT)])
    def test_plane_lump(self, normal, dist):
        plane = Plane()
        plane.dist = dist
        plane.normal = normal

        assert all([a == b for a, b in zip(plane.normal, normal)])
        assert plane.dist == dist

    def teardown(self):
        pass
