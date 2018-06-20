from setuptools import setup, find_packages
import os


NAME = "joint-vector"
ROOT_DIR = os.path.abspath(os.path.dirname(__file__))

setup(
    name=NAME,
    description="",
    license="MIT",
    packages=find_packages(exclude=["test"]),
    install_requires=[
        "Keras",
        "mxnet",
        "pydash",
        "spacy",
        "tensorflow-gpu"
    ],
    test_suite="test"
)
