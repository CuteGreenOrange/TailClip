import os
import numpy as np
import torch
import csv
import argparse
from models.IFRNet_S import Model
from utils import read
import imageio
import torch.nn.functional as F
import torch
import time


INDEPENDENT_PRUNE_FLAG = False
# DEVICE = torch.device("cuda" if args.gpu_no >= 0 else "cpu")
DEVICE = torch.device("cuda")
CONFIG = {
    "encoder.pyramid1.0.0":(3,24),
    "encoder.pyramid1.1.0":(24,24),
    "encoder.pyramid2.0.0":(24,36),
    "encoder.pyramid2.1.0":(36,36),
    "encoder.pyramid3.0.0":(36,54),
    "encoder.pyramid3.1.0":(54,54),
    "encoder.pyramid4.0.0":(54,72),
    "encoder.pyramid4.1.0":(72,72),

    "decoder4.convblock.0.0":(145,144),
    "decoder4.convblock.1.conv1.0":(144,144),
    "decoder4.convblock.1.conv2.0":(24,24),
    "decoder4.convblock.1.conv3.0":(144,144),
    "decoder4.convblock.1.conv4.0":(24,24),
    "decoder4.convblock.1.conv5":(144,144),
    "decoder4.convblock.2":(144,58),

    "decoder3.convblock.0.0":(166,162),
    "decoder3.convblock.1.conv1.0":(162,162),
    "decoder3.convblock.1.conv2.0":(24,24),
    "decoder3.convblock.1.conv3.0":(162,162),
    "decoder3.convblock.1.conv4.0":(24,24),
    "decoder3.convblock.1.conv5":(162,162),
    "decoder3.convblock.2":(162,40),

    "decoder2.convblock.0.0":(112,108),
    "decoder2.convblock.1.conv1.0":(108,108),
    "decoder2.convblock.1.conv2.0":(24,24),
    "decoder2.convblock.1.conv3.0":(108,108),
    "decoder2.convblock.1.conv4.0":(24,24),
    "decoder2.convblock.1.conv5":(108,108),
    "decoder2.convblock.2":(108,28),

    "decoder1.convblock.0.0":(76,72),
    "decoder1.convblock.1.conv1.0":(72,72),
    "decoder1.convblock.1.conv2.0":(24,24),
    "decoder1.convblock.1.conv3.0":(72,72),
    "decoder1.convblock.1.conv4.0":(24,24),
    "decoder1.convblock.1.conv5":(72,72),
    "decoder1.convblock.2":(72,8),
}

SIDE_CHANNEL = [
    "decoder4.convblock.1.conv2.0",
    "decoder4.convblock.1.conv2.1",
    "decoder4.convblock.1.conv4.0",
    "decoder4.convblock.1.conv4.1",

    "decoder3.convblock.1.conv2.0",
    "decoder3.convblock.1.conv2.1",
    "decoder3.convblock.1.conv4.0",
    "decoder3.convblock.1.conv4.1",

    "decoder2.convblock.1.conv2.0",
    "decoder2.convblock.1.conv2.1",
    "decoder2.convblock.1.conv4.0",
    "decoder2.convblock.1.conv4.1",

    "decoder1.convblock.1.conv2.0",
    "decoder1.convblock.1.conv2.1",
    "decoder1.convblock.1.conv4.0",
    "decoder1.convblock.1.conv4.1"
]

BEGIN_PYRAMID = [
    "decoder1.convblock.0.0",
    "decoder2.convblock.0.0",
    "decoder3.convblock.0.0",
]

PRUNE_LAYERS=[
    ["encoder.pyramid1.0.0"],
    ["encoder.pyramid2.0.0"],
    ["encoder.pyramid3.0.0"],
    ["encoder.pyramid4.0.0"],
    ["decoder4.convblock.0.0", "decoder4.convblock.1.conv5"],
    ["decoder4.convblock.1.conv3.0"],
    ["decoder3.convblock.0.0", "decoder3.convblock.1.conv5"],
    ["decoder3.convblock.1.conv3.0"],
    ["decoder2.convblock.0.0", "decoder2.convblock.1.conv5"],
    ["decoder2.convblock.1.conv3.0"],
    ["decoder1.convblock.0.0", "decoder1.convblock.1.conv5"],
    ["decoder1.convblock.1.conv3.0"],
]
FIRST_DECODER = 'decoder4.convblock.0.0'   

PYRAMID_PRUNE = True
LAYER_PRUNE = False
PYRAMID = 1
LAST_ENCODER_CONV = None
PRUNE_PERCENTAGE = 0.7
pyramid_layer_number_old = None
PRUNE_CONFIGS = {}
END_PYRAMID = [
    "decoder4.convblock.2",
    "decoder3.convblock.2",
    "decoder2.convblock.2",
    "decoder1.convblock.2",
]

def gen_config(layerIds):
    prune_config={}
    for layerId in layerIds:
        if layerId.startswith('encoder'):
            prune_config[layerId]=int(CONFIG[layerId][1]*PRUNE_PERCENTAGE)
        else:
            prune_config[layerId]=min(int(CONFIG[layerId][1]*PRUNE_PERCENTAGE),CONFIG[layerId][1]-24)
    return prune_config


def gen_pyramidconfig():
    global LAST_ENCODER_CONV
    global pyramid_layer_number_old
    LAST_ENCODER_CONV = "encoder.pyramid{}.1.0".format(PYRAMID)
    pyramid_layer_number_old = CONFIG["encoder.pyramid{}.1.0".format(PYRAMID)][0]
    pyramid_layer_number = int(pyramid_layer_number_old*(1-PRUNE_PERCENTAGE))
    prune_config={}
    prune_config = {
        "encoder.pyramid{}.0.0".format(PYRAMID):CONFIG["encoder.pyramid{}.0.0".format(PYRAMID)][1]-pyramid_layer_number,
        "encoder.pyramid{}.1.0".format(PYRAMID):CONFIG["encoder.pyramid{}.1.0".format(PYRAMID)][1]-pyramid_layer_number, 
        "decoder{}.convblock.0.0".format(PYRAMID):CONFIG["decoder{}.convblock.0.0".format(PYRAMID)][1]-pyramid_layer_number*3,
        "decoder{}.convblock.1.conv1.0".format(PYRAMID):min(CONFIG["decoder{}.convblock.1.conv1.0".format(PYRAMID)][0]-pyramid_layer_number*3,CONFIG["decoder{}.convblock.1.conv1.0".format(PYRAMID)][0]-24),
        "decoder{}.convblock.1.conv3.0".format(PYRAMID):min(CONFIG["decoder{}.convblock.1.conv3.0".format(PYRAMID)][0]-pyramid_layer_number*3,CONFIG["decoder{}.convblock.1.conv3.0".format(PYRAMID)][0]-24),
        "decoder{}.convblock.1.conv5".format(PYRAMID):CONFIG["decoder{}.convblock.1.conv5".format(PYRAMID)][0]-pyramid_layer_number*3
    }
    for config in prune_config:
        if prune_config[config]<0:
            print("prune_config[config]<0")
            prune_config[config]=0
    if PYRAMID != 4:
        prune_config["decoder{}.convblock.2".format(PYRAMID+1)]=CONFIG["decoder{}.convblock.2".format(PYRAMID+1)][1]-4-pyramid_layer_number
    return prune_config

def prune_network(network=None):
    network = prune_step(network, PRUNE_CONFIGS, INDEPENDENT_PRUNE_FLAG)
    network = network.to(DEVICE)

    return network

def set_submodule(network, path, new_module):


    parts = path.split('.')
    current_module = network

    for idx in range(len(parts)-1):
        current_module = getattr(current_module, parts[idx])
    last_part = parts[-1]
    setattr(current_module, last_part, new_module)

def prune_step(network, prune_configs, independent_prune_flag):
    network = network.cpu()
    dim = 0 
    residue = None
    if PYRAMID_PRUNE == True:
        encoder_channel_index = None

    for name, module in network.named_modules():
        if name in SIDE_CHANNEL:
            continue
        if isinstance(module, torch.nn.Conv2d) or isinstance (module, torch.nn.ConvTranspose2d):
            if dim == 1:
                if isinstance(module, torch.nn.Conv2d):
                    if name in BEGIN_PYRAMID: 
                        assert PYRAMID_PRUNE
                        new_channel_index = encoder_channel_index+ [x - 4 + pyramid_layer_number_old for x in channel_index] + [x - 4 + pyramid_layer_number_old*2 for x in channel_index]
                        new_, residue = get_new_conv(module, dim, new_channel_index, independent_prune_flag)
                    elif name == FIRST_DECODER:
                        assert PYRAMID_PRUNE
                        new_channel_index = encoder_channel_index+ [x + pyramid_layer_number_old  for x in encoder_channel_index]
                        new_, residue = get_new_conv(module, dim, new_channel_index, independent_prune_flag)
                    else:
                        new_, residue = get_new_conv(module, dim, channel_index,    independent_prune_flag)
                elif isinstance(module, torch.nn.ConvTranspose2d):
                    new_, residue = get_new_convT(module, dim, channel_index, independent_prune_flag)
                set_submodule(network, name, new_)
                dim ^= 1
                module = new_
            if name in prune_configs: 
                if isinstance(module, torch.nn.Conv2d):
                    channel_index = get_channel_index(module.weight.data, prune_configs[name], residue)
                    if PYRAMID_PRUNE and name == LAST_ENCODER_CONV: 
                        encoder_channel_index = channel_index
                    new_ = get_new_conv(module, dim, channel_index, independent_prune_flag)
                elif isinstance(module, torch.nn.ConvTranspose2d):
                    
                    channel_index = get_channel_index_convT(module.weight.data, prune_configs[name], residue)
                    new_ = get_new_convT(module, dim, channel_index, independent_prune_flag)
                set_submodule(network, name, new_)
                dim ^= 1
            else:
                residue = None
        elif dim == 1 and isinstance(module, torch.nn.PReLU):
            new_ = get_new_prelu(module, channel_index)
            set_submodule(network, name, new_)
            module = new_
    

    return network


def get_channel_index(kernel, num_elimination, residue=None):

    sum_of_kernel = torch.sum(torch.abs(kernel.view(kernel.size(0), -1)), dim=1)
    if residue is not None:
        sum_of_kernel += torch.sum(torch.abs(residue.view(residue.size(0), -1)), dim=1)
    
    vals, args = torch.sort(sum_of_kernel)

    return args[:num_elimination].tolist()


def get_channel_index_convT(kernel, num_elimination, residue=None):
    kernel_1 = kernel[:,4:,:,:]   
    sum_of_kernel = torch.sum(torch.abs(kernel_1), dim=[0,2,3])
    if residue is not None:
        sum_of_kernel += torch.sum(torch.abs(residue), dim=[0,2,3])
    
    vals, args = torch.sort(sum_of_kernel)

    index_list = args[:num_elimination].tolist()
    index_list = [x + 4 for x in index_list]
    return index_list


def index_remove(tensor, dim, index, removed=False):
    if tensor.is_cuda:
        tensor = tensor.cpu()
    size_ = list(tensor.size())
    new_size = tensor.size(dim) - len(index)
    size_[dim] = new_size
    new_size = size_

    select_index = list(set(range(tensor.size(dim))) - set(index))
    new_tensor = torch.index_select(tensor, dim, torch.tensor(select_index))

    if removed:
        return new_tensor, torch.index_select(tensor, dim, torch.tensor(index))

    return new_tensor

def get_new_conv(conv, dim, channel_index, independent_prune_flag=False):
    if dim == 0:
        new_conv = torch.nn.Conv2d(
            in_channels=conv.in_channels,
            out_channels=int(conv.out_channels - len(channel_index)),
            kernel_size=conv.kernel_size,
            stride=conv.stride, 
            padding=conv.padding, 
            dilation=conv.dilation
        )
        new_conv.weight.data = index_remove(conv.weight.data, dim, channel_index)
        new_conv.bias.data = index_remove(conv.bias.data, dim, channel_index)
        return new_conv

    elif dim == 1:
        new_conv = torch.nn.Conv2d(
            in_channels=int(conv.in_channels - len(channel_index)),
            out_channels=conv.out_channels,
            kernel_size=conv.kernel_size,
            stride=conv.stride, 
            padding=conv.padding, 
            dilation=conv.dilation
        )
        new_weight = index_remove(conv.weight.data, dim, channel_index, independent_prune_flag)
        residue = None
        if independent_prune_flag:
            new_weight, residue = new_weight
        new_conv.weight.data = new_weight
        new_conv.bias.data = conv.bias.data   
        return new_conv, residue    
    
def get_new_convT(convT, dim, channel_index, independent_prune_flag=False):
    if dim == 0:
        new_convT = torch.nn.ConvTranspose2d(
            in_channels=convT.in_channels,
            out_channels=int(convT.out_channels- len(channel_index)),
            kernel_size=convT.kernel_size,
            stride=convT.stride,
            padding=convT.padding, 
            dilation=convT.dilation
        )
        new_convT.weight.data = index_remove(convT.weight.data, 1, channel_index)
        new_convT.bias.data = index_remove(convT.bias.data, dim, channel_index)
        return new_convT

    elif dim == 1:
        new_convT = torch.nn.ConvTranspose2d(
            in_channels=int(convT.in_channels - len(channel_index)),
            out_channels=convT.out_channels,
            kernel_size=convT.kernel_size,
            stride=convT.stride,
            padding=convT.padding, 
            dilation=convT.dilation
        )
        new_weight = index_remove(convT.weight.data, 0, channel_index, independent_prune_flag)
        residue = None
        if independent_prune_flag:
            new_weight, residue = new_weight
        new_convT.weight.data = new_weight
        new_convT.bias.data = convT.bias.data

        return new_convT, residue 

def get_new_prelu(prelu, channel_index):
    new_prelu = torch.nn.PReLU(num_parameters=int(prelu.num_parameters - len(channel_index)))
    new_prelu.weight.data = index_remove(prelu.weight.data, 0, channel_index)
    return new_prelu

def get_new_linear(linear, channel_index):
    new_linear = torch.nn.Linear(in_features=int(linear.in_features - len(channel_index)),
                                out_features=linear.out_features,
                                bias=linear.bias is not None)
    new_linear.weight.data = index_remove(linear.weight.data, 1, channel_index)
    new_linear.bias.data = linear.bias.data
    
    return new_linear

def infer_test(model):
    model = prune_network(model)

    if HALF:
        model = model.half().to(DEVICE)
        img1 = torch.rand(1, 3, 720, 1280).half().to(DEVICE)
        img2 = torch.rand(1, 3, 720, 1280).half().to(DEVICE)
        embt = torch.rand(1,1,1,1).half().to(DEVICE)
    else:
        model = model.to(DEVICE)
        img1 = torch.rand(1, 3, 720, 1280).to(DEVICE)
        img2 = torch.rand(1, 3, 720, 1280).to(DEVICE)
        embt = torch.rand(1,1,1,1).to(DEVICE)
    for i in range(10):
        imgt_pred = model.inference(img1, img2, embt)
    torch.cuda.synchronize()
    start_time = time.time()
    imgt_pred = model.inference(img1, img2, embt)
    torch.cuda.synchronize()
    end_time = time.time()
    print("only_infer_time:", end_time-start_time)  
    return end_time-start_time

if __name__ == "__main__":
    HALF= True
    PYRAMID_PRUNE = False
    LAYER_PRUNE = False
    PYRAMID = 3
    PRUNE_PERCENTAGE = 0.9
    parser = argparse.ArgumentParser()
    parser.add_argument('--prune_mode', default='PYRAMID_PRUNE', type=str, help='PYRAMID_PRUNE, LAYER_PRUNE')
    parser.add_argument('--pyramid', default=4, type=int)
    parser.add_argument('--layer', default=1, type=int, help="  1:  [encoder.pyramid1.0.0],  2: [encoder.pyramid2.0.0], \
                        3: [encoder.pyramid3.0.0], 4: [encoder.pyramid4.0.0], \
                        5: [decoder4.convblock.0.0, decoder4.convblock.1.conv5], 6: [decoder4.convblock.1.conv3.0], \
                        7: [decoder3.convblock.0.0, decoder3.convblock.1.conv5], 8: [decoder3.convblock.1.conv3.0], \
                        9: [decoder2.convblock.0.0, decoder2.convblock.1.conv5], 10: [decoder2.convblock.1.conv3.0], \
                        11: [decoder1.convblock.0.0, decoder1.convblock.1.conv5], 12: [decoder1.convblock.1.conv3.0]")
    parser.add_argument('--prune_percent', default=0.3, type=float)
    parser.add_argument('--output_file', default="output.csv", type=str)
    args = parser.parse_args()
    if args.prune_mode =="PYRAMID_PRUNE":
        PYRAMID_PRUNE = True
        LAYER_PRUNE = False
        print("prune_mode:{}, pyramid:{}, percent:{}".format(args.prune_mode,args.pyramid,args.prune_percent))
    elif args.prune_mode =="LAYER_PRUNE":
        PYRAMID_PRUNE = False
        LAYER_PRUNE = True
        print("prune_mode:{}, layer:{}, percent:{}".format(args.prune_mode,args.layer,args.prune_percent))
    PYRAMID = args.pyramid
    PRUNE_PERCENTAGE = args.prune_percent
    output_csv_file = args.output_file

    with torch.no_grad():

        model = Model(config=CONFIG).eval()
        model.load_state_dict(torch.load('./checkpoint/IFRNet_S.pth'))
        if PYRAMID_PRUNE==True:
            PRUNE_CONFIGS = gen_pyramidconfig()
            print("PRUNE_CONFIGS:{}".format(PRUNE_CONFIGS))
            infertime = infer_test(model)
            out=[args.prune_mode,PYRAMID,PRUNE_PERCENTAGE,infertime]
            print("out:{}".format(out))
            with open(output_csv_file, mode='a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(out)
        if LAYER_PRUNE == True:
            layerids = PRUNE_LAYERS[args.layer]
            PRUNE_CONFIGS = gen_config(layerids)
            print("PRUNE_CONFIGS:{}".format(PRUNE_CONFIGS))
            infertime = infer_test(model)
            out=[args.prune_mode,args.layer,PRUNE_PERCENTAGE,infertime]
            print("out:{}".format(out))
            with open(output_csv_file, mode='a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(out)