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
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3 :: Only',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: Implementation :: CPython',
        'Topic :: Software Development :: Libraries',
        'Topic :: Software Development :: Libraries :: Python Modules',
        'Typing :: Typed'

    ],
    install_requires=[
        'numpy~=1.23.2'
    ],
    extras_require={
        'dev': [
            'pytest~=7.1.2',
            'pytest-cov~=3.0.0'
        ]
    }
)
