import argparse
import torch
import time 

from models.hardnext import HarDNeXt
from torchstat import stat


def main():
    arch = [28, 32, 39, 50, 56]
    parser = argparse.ArgumentParser(description="Pytorch model summary. ")

    parser.add_argument(
        '-a',
        '--arch',
        default=32,
        choices = arch,
        type=int

    )

    parser.add_argument(
        '-he',
        '--height',
        default=224,
        type=int
    )

    parser.add_argument(
        '-w',
        '--width',
        default=224,
        type=int
    )

    parser.add_argument(
        '--gpu',
        default=0,
        type=int
    )

    args=parser.parse_args()

    model = HarDNeXt(arch=args.arch)
    # print(model)
    # print("test output")
    arch = int(args.arch) 
    full_model_name = ''

    
    full_model_name = "HarDNeXt("+str(args.arch)+')'
    stat(model, (3, args.height, args.width))
    print("Target model: "+full_model_name)


if __name__ == '__main__':
    main()