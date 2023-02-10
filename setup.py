import os
from setuptools import setup, find_packages

module_name = "qpert"

file_dir = os.path.dirname(os.path.realpath(__file__))
absdir = lambda p: os.path.join(file_dir, p)

############### versioning ###############
verfile = os.path.abspath(os.path.join(module_name, "_version.py"))
version = {"__file__": verfile}
with open(verfile, "r") as fp:
    exec(fp.read(), version)

setup(
    name=module_name,
    version=version["get_version"](),
    description='Supervised learning DENSE framework',
    url='https://github.com/machine-discovery/sldense/',
    author='mfkasim1',
    author_email='firman.kasim@gmail.com',
    license='Commercial',
    packages=find_packages(),
    python_requires=">=3.7",
    install_requires=[
        "numpy>=1.8.2",
        "scipy>=1.7.2",
        "torch>=1.11, !=1.12.0",  # https://github.com/pytorch/pytorch/issues/80489 because 1.12.0 do not work well with functorch
        "tensorboard>=2.2.0",
        "matplotlib>=3.4.2",
    ],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Physics",
        "Topic :: Scientific/Engineering :: Mathematics",
        "License :: OSI Approved :: MIT License",

        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
    ],
    keywords="project deep-learning",
    zip_safe=False
)
