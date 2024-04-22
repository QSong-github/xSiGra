import argparse

from train_nanostring import train_nano_fov
from train_visium import train_10x

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset", type=str, default="nanostring", help="should be nanostring or 10x"
    )
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--root", type=str, default="../dataset/nanostring")
    parser.add_argument("--epochs", type=int, default=2000)
    parser.add_argument("--id", type=str, default="fov1")
    parser.add_argument("--img_name", type=str, default="F001")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument(
        "--save_path", type=str, default="../checkpoint/nanostring_final"
    )
    parser.add_argument("--ncluster", type=int, default=8)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--use_gray", type=float, default=0)
    parser.add_argument("--test_only", type=int, default=0)
    parser.add_argument("--pretrain", type=str, default="final_0.pth")
    parser.add_argument(
        "--cluster_method", type=str, default="leiden", help="leiden or mclust"
    )
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--img_size", type=str, default="50,50")
    parser.add_argument("--neurons", type=str, default="512,30")
    parser.add_argument("--num_layers", type=str, default="2")

    opt = parser.parse_args()

    if opt.dataset == "nanostring":
        train_nano_fov(opt)

    elif opt.dataset == "10x":
        train_10x(opt)
