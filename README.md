# IBSPLib
![Tests](https://github.com/Mix-Anik/ibsplib/actions/workflows/test.yml/badge.svg)
![pypi](https://img.shields.io/pypi/v/ibsplib.svg)
![python](https://img.shields.io/pypi/pyversions/ibsplib.svg)

ibsplib is Python package for working with Quake 3 IBSP structures  
References were taken from http://www.mralligator.com/q3

- Parsing IBSP
- Typings & code completion included
- Numpy friendly
- More to come....

### Installation
Available on PyPI, just:
```sh
pip install ibsplib
```

### Usage
```py
from ibsplib import IBSP


bsp_path = '...\\<map name>.bsp'
bsp = IBSP(bsp_path)

print(f'Version: {bsp.header.version}')
print('Textures used:')

for tex in bsp.textures:
    print(f'-\t{tex.name}')
```

## License
MIT
