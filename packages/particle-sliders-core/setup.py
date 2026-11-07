"""Keep native product implementations out of wheels and strict editables."""
from setuptools import setup
from _core_build import CoreBuildPy

setup(cmdclass={'build_py': CoreBuildPy})
