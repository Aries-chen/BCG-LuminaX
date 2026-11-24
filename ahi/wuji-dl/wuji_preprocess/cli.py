import fire
import sys

from wuji_preprocess.runner import preprocess


def main():
    sys.path.append(".")
    fire.Fire(preprocess)
