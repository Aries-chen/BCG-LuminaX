import fire
import sys

from wuji_index.runner import build_index


def main():
    sys.path.append(".")
    fire.Fire(build_index)
