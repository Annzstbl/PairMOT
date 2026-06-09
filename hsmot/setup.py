from setuptools import find_packages, setup

setup(
    name="hsmot",
    version="0.3.0",
    author="LTH",
    author_email="lth@163.com",
    description="HSMOT preprocessing, data pipelines and evaluation utilities",
    long_description_content_type="text/markdown",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.19.0,<2.0.0",
        "torch>=1.8.0",
        "opencv-python>=4.5.0",
        "scipy",
        "scikit-learn",
        "tqdm",
    ],
    extras_require={
        "dev": ["pytest", "ruff"],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
)
