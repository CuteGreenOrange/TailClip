import os
import numpy as np
import torch
import torch.nn.functional as F
from liteflownet.run import estimate
from utils import read, write

data_dir = '/home/Videos/test'

sequences_dir = os.path.join(data_dir, 'seqs')
flow_dir = os.path.join(data_dir, 'flow')

if not os.path.exists(flow_dir):
    os.makedirs(flow_dir)



def pred_flow(img1, img2):
    img1 = torch.from_numpy(img1).float().permute(2, 0, 1) / 255.0
    img2 = torch.from_numpy(img2).float().permute(2, 0, 1) / 255.0

    flow = estimate(img1, img2)

    flow = flow.permute(1, 2, 0).cpu().numpy()
    return flow

if __name__ == "__main__":
    image_paths = sorted(os.listdir(sequences_dir))
    for i in range(len(image_paths)-3):
        img0_path = os.path.join(sequences_dir, image_paths[i])
        imgt_path = os.path.join(sequences_dir, image_paths[i+3])
        img1_path = os.path.join(sequences_dir, image_paths[i+2])
        flow_t0_path = os.path.join(flow_dir, image_paths[i+3].split(".")[0]+"_t0.flo")
        flow_t1_path = os.path.join(flow_dir, image_paths[i+3].split(".")[0]+"_t1.flo")

        img0 = read(img0_path)
        imgt = read(imgt_path)
        img1 = read(img1_path)

        flow_t0 = pred_flow(imgt, img0)
        flow_t1 = pred_flow(imgt, img1)

        write(flow_t0_path, flow_t0)
        write(flow_t1_path, flow_t1)


