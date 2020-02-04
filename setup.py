#!/usr/bin/env python3
from setuptools import setup

setup(
    name="featurevis",
    version="0.2.1",
    description="Visualize features cells are responsive to via gradient ascent",
    author="Erick Cobos",
    author_email="ecobos@bcm.edu",
    license="MIT",
    url="https://github.com/cajal/featurevis",
    keywords="feature visualization MEI pytorch",
    packages=["featurevis"],
    install_requires=["torch>=0.4.0", "scipy", "numpy", "torchvision", "datajoint", "nnfabrik"],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3 :: Only",
        "License :: OSI Approved :: MIT License",
        "Natural Language :: English" "Topic :: Scientific/Engineering :: Bio-Informatics",
    ],
)
