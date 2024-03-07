#!/usr/bin/env python3
from setuptools import setup

setup(
    name="mei",
    version="0.0.0",
    description="A framework used to generate most exciting images (MEIs)",
    author="Christoph Blessing",
    author_email="chris24.blessing@gmail.com",
    license="MIT",
    url="https://github.com/sinzlab/mei",
    keywords="feature visualization MEI pytorch",
    python_requires="<3.10",
    packages=["mei"],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3 :: Only",
        "License :: OSI Approved :: MIT License",
        "Natural Language :: English" "Topic :: Scientific/Engineering :: Bio-Informatics",
    ],
)
