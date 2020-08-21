from distutils.core import setup, Extension
from Cython.Distutils import build_ext
from Cython.Build import cythonize
import numpy as np
import os
os.environ["CC"] = "g++"
#os.environ["CXX"] = "g++"

compile_args=['-O3','-fopenmp','-ffast-math','-std=c++11']
link_args=['-fopenmp']

ext_modules = [
     Extension("la",
               sources=["./src/la.pyx"],
               extra_compile_args=compile_args, extra_link_args=link_args, language="c++"),
     Extension("uwater",
               sources=["./src/uwater.pyx"],
               depends=["./src/tools.h"],
               extra_compile_args=compile_args, extra_link_args=link_args, language="c++"),
     Extension("water",  
               sources=["./src/water.pyx", "./src/cwater.cpp", "./src/tools.cpp"], 
               #include_dirs=["./include"],
               depends=["./src/tools.h"], 
               extra_compile_args=compile_args, extra_link_args=link_args, language="c++")
  ]

setup(
   cmdclass = {'build_ext': build_ext},
   ext_modules=cythonize(ext_modules, language_level=3, annotate=True),
   include_dirs=[np.get_include(),"."]
)

