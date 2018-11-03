from setuptools import setup, find_packages


NAME = "jointvector"

setup(
    name=NAME,
    version="1.0.0",
    description="",
    license="MIT",
    packages=find_packages(exclude=["test"]) + ["props"],
    include_package_data=True,
    extras_require={
        "tensorflow-gpu": ["tensorflow-gpu"]
    },
    install_requires=[
        "Cython",
        "fasttextmirror",
        "Keras",
        "matplotlib",
        "mxnet",
        "nltk",
        "nose",
        "numpy",
        "opencv-python",
        "pycocotools",
        "pydash",
        "pytest",
        "scikit-image",
        "tensorflow"
    ],
    test_suite="test"
)
