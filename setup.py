import os
from setuptools import setup, find_packages

module_name = "deer"

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
    description='Parallelizing RNN and NeuralODE models',
    url='https://github.com/machine-discovery/deer/',
    author='mfkasim1',
    author_email='muhammad@machine-discovery.com',
    license='BSD-3',
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "jax>=0.4.28",
        "numpy>=1.24.0",
        "scipy>=1.10.1",
        "equinox>=0.10.6",
        "matplotlib>=3.6.2",
    ],
    extras_require={
        "replication": [
            "tensorboard>=2.13.0",
            "flax>=0.7.0",
            "ml_dtypes>=0.2.0",
            "pytorch_lightning>=2.0.1",
            "tensorboardX>=2.6.1",
            "tqdm>=4.66.1",
            "pytest>=7.4.2",
        ],
        "dev": [
            "pytest>=7.4.2",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Physics",
        "Topic :: Scientific/Engineering :: Mathematics",
        "License :: OSI Approved :: BSD-3 License",

        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    keywords="deep-learning rnn neural-networks neural-ode recurrent-neural-networks parallelization",
    zip_safe=False
)
