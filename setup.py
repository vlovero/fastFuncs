import os
import sys
import platform
import subprocess
import numpy as np
from setuptools import setup, Extension, find_packages
from setuptools.command.install import install as _install


BASE_DIR = os.path.dirname(__file__)


class install(_install):
    def run(self):
        cmd = [sys.executable, __file__, "build"]
        process = subprocess.Popen(cmd)
        process.wait()
        _install.run(self)


if __name__ == "__main__":
    if platform.system() == "Darwin":
        compile_args = ["-std=c++17", "-march=native", "-Ofast", "-Xpreprocessor", "-fopenmp"]
    else:
        compile_args = ["-std=c++17", "-march=native", "-Ofast", "-fopenmp"]

    module = Extension("fastfuncs._fastfuncs",
                       sources=["fastfuncs/src/_fastfuncs.cpp"],
                       include_dirs=["fastfuncs/include", np.get_include()],
                       extra_compile_args=compile_args,
                       extra_link_args=["-lomp"])

    setup(name="fastfuncs",
          packages=find_packages(),
          version="0.1",
          author="Vincent Lovero",
          author_email="vllovero@ucdavis.edu",
          description="faster ufuncs that have option to be parallelized",
          install_requires=["setuptools", "numpy>=1.22"],
          cmdclass={'install': install},
          include_package_data=True,
          package_data={'fastfuncs': ['py.typed', '__init__.pyi'],
                        'fastfuncs.parallel': ['py.typed', '__init__.pyi']},
          classifiers=["Development Status :: 4 - Beta",
                       "Operating System :: MacOS",
                       "License :: OSI Approved :: MIT License",
                       "Programming Language :: C++",
                       "Programming Language :: Python :: 3.6"
                       "Programming Language :: Python :: 3.7"
                       "Programming Language :: Python :: 3.8"
                       "Programming Language :: Python :: 3.9"
                       "Programming Language :: Python :: 3.10"],
          ext_modules=[module])
