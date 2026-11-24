from setuptools import setup, find_namespace_packages

VERSION = "0.5.1"
DESCRIPTION = (
    "WujiDL: A simple deep learning framework based on Lightning and Weights & Biases."
)
LONG_DESCRIPTION = (
    "WujiDL: A simple deep learning framework based on Lightning and Weights & Biases."
)

setup(
    name="wuji-dl",
    version=VERSION,
    author="Yuxin Chen",
    author_email="itsme_yxchen@outlook.com",
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    packages=find_namespace_packages(include=("wuji_*",)),
    install_requires=[
        "astropy<6",
        "fire",
        "hrv-analysis",
        "lightning",
        "omegaconf",
        "pandas",
        "rich",
        "torchmetrics",
        "wandb<0.18",
    ],
    keywords=["python", "deep learning"],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Programming Language :: Python :: 3",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: Microsoft :: Windows",
    ],
    entry_points={
        "console_scripts": [
            "wuji-dl = wuji_dl.cli:main",
            "wuji-fit = wuji_dl.cli:fit",
            "wuji-test = wuji_dl.cli:test",
            "wuji-preprocess = wuji_preprocess.cli:main",
            "wuji-index = wuji_index.cli:main",
            "wuji-export-onnx = wuji_dl.cli:export_onnx"
        ]
    },
)
