import argparse
from typing_extensions import IntVar
import torch
import time 

from models.hardnext import HarDNeXt
from models.resnet import resnet50

def speed_test(full_model_name, model, height, width, batch,  gpu):

    # start = torch.cuda.Event(enable_timing=True)
    # end = torch.cuda.Event(enable_timing=True)
    torch.cuda.set_device(gpu)
    inference_time_list = []
    model.eval().cuda()

    for i in range(50):
        #model.eval().cuda()
        #model.eval()
        input = torch.randn(batch, 3, height, width).cuda()

        start_time = time.time()

        with torch.no_grad():
            model(input)
        # Waits for everything to finish running
        torch.cuda.synchronize()
        end_time = time.time()

        if i>10:
            inference_time_list.append((end_time - start_time)/batch)

    avg_FPS = 1/(sum(inference_time_list)/len(inference_time_list))
    
    gpu_name = torch.cuda.get_device_name(gpu)
    print("Measuring FPS completed !!!")
    print("{:s} FPS on {:s} ({:d} x {:d} )(batch size {:d}): {:.1f}".format(full_model_name, gpu_name, height, width, batch, avg_FPS))
    
    return 
    #return avg_FPS

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
        '-b',
        '--batch',
        default=1,
        type=int
    )

    parser.add_argument(
        '--gpu',
        default=0,
        type=int
    )

    args=parser.parse_args()

    model = HarDNeXt(arch=args.arch)
    #model = resnet50()
    # print(model)
    # print("test output")
    arch = int(args.arch) 
    full_model_name = ''

    # if args.depthwise:
    #     full_model_name = "HarDNeXt("+str(args.arch)+"DS)"
    # else:
    #     full_model_name = "HarDNeXt("+str(args.arch)+")"
    full_model_name = "HarDNeXt("+str(args.arch)+')'

    print("Measuring FPS ...")
    speed_test(full_model_name, model, args.height, args.width, args.batch, args.gpu)


if __name__ == '__main__':
    main()
