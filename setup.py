"""
    This is the setup file for the PIP package.

    Copyright 2021 Nathan Nesbitt
"""

from setuptools import setup, find_packages

from os import path

this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, "README.md"), encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="codesummary",
    version="1.0.0",
    description="A deployable REST API for NLP Code Summarization.",
    url="https://github.com/Nathan-Nesbitt/CodeSummary",
    author=("Nathan Nesbitt"),
    packages=find_packages(),
    install_requires=[
        "Flask",
        "flask-cors",
        "werkzeug",
        "torch==1.8.1",
        "rouge",
        "torchtext==0.6",
        "matplotlib",
        "nltk",
        "javalang",
        "tqdm",
        "spiral @ git+https://github.com/casics/spiral.git"
    ],
    zip_safe=False,
    long_description=long_description,
    long_description_content_type="text/markdown",
)