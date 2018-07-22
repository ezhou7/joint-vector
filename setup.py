from setuptools import setup, find_packages
import os


NAME = "joint-vector"

setup(
    name=NAME,
    description="",
    license="MIT",
    packages=find_packages(exclude=["test"]) + ["props"],
    include_package_data=True,
    setup_requires=["Cython"],
    extras_require={
        "tensorflow-gpu": ["tensorflow-gpu"]
    },
    install_requires=[
        "Cython",
        "Keras",
        "mxnet",
        "nose",
        "numpy",
        "pydash",
        "pytest",
        "spacy",
        "tensorflow"
    ],
    test_suite="test"
)
