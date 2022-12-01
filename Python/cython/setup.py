from setuptools import setup
from Cython.Build import cythonize
from distutils.extension import Extension

extensions = [
    Extension("kvstore", sources=["kvstore.pyx"],
              extra_compile_args=["-O3"], language="c++")
]

setup(
    name="kvstore",
    ext_modules=cythonize(extensions)
)
