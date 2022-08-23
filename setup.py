from setuptools import setup


setup(
    name='ibsplib',
    version='0.2.0',
    author='Mix-Anik',
    description='Python library for working with Quake 3 IBSP structures',
    packages=['ibsplib'],
    license='MIT',
    url='https://github.com/Mix-Anik/ibsplib',
    keywords=['quake 3', 'q3', 'IBSP', 'BSP'],
    install_requires=[
        'numpy~=1.23.2',
        'pytest~=7.1.2'
    ]
)
