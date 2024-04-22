import argparse

from processing_nanostring import processing_nano
from processing_visium import processing_10x

parser = argparse.ArgumentParser()
parser.add_argument(
    "--dataset",
    type=str,
    default="nanostring",
    help="should be nanostring, merscope or 10x",
)
opt = parser.parse_args()

if opt.dataset == "nanostring":
    processing_nano()
elif opt.dataset == "10x":
    processing_10x()
