from setuptools import setup
from torch.utils import cpp_extension

setup(name='kvstore',
      ext_modules=[cpp_extension.CppExtension('kvstore', ['kvstore.cc'],
                                              extra_compile_args=['-fopenmp', '-march=native'])],
      cmdclass={'build_ext': cpp_extension.BuildExtension})
