from setuptools import find_packages, setup

from pargraph.about import __version__

with open("requirements.txt", "rt") as f:
    requirements = [i.strip() for i in f.readlines()]

setup(
    name="pargraph",
    version=__version__,
    packages=find_packages(exclude=("tests",)),
    install_requires=requirements,
    extras_require={"graphblas": ["python-graphblas", "numpy"]},
    description="Pargraph parallel graph computation library",
)
