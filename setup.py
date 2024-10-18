from setuptools import setup, find_packages, Extension

from Cython.Build import cythonize

ext_modules = [
    Extension(
        name="pathfinding.cpf",
        sources=["pathfinding/wrapper.pyx"],
        language="c++",
        extra_compile_args=["-std=c++17"],
    )
]

setup(
    name="pathfinding",
    description="Implementation of some pathfinding algorithms",
    version="0.0.1",
    packages=find_packages(),
    ext_modules=cythonize(ext_modules),
)
