# IBSPLib

ibsplib is Python package for working with Quake 3 IBSP structures
References were taken from http://www.mralligator.com/q3

- Parsing IBSP
- No dependencies
- Typings & code completion included
- More to come....

### Installation
Available on PyPI, just:
```sh
pip install ibsplib
```

### Usage
```py
from pathlib import Path
from ibsplib import IBSP


bsp_path = f'{Path(__file__).parent}\\<map name>.bsp'

with open(bsp_path, 'rb') as f:
    bsp_buffer = bytearray(f.read())

bsp = IBSP(bsp_buffer)

print(f'Version: {bsp.header.version}')
print('Textures used:')

for tex in bsp.textures:
    print(f'-\t{tex.name}')
```

## License
MIT
