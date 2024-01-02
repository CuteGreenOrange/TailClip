import os
import math
import time
import random
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from datasets import Pre_Train_Dataset, Pre_Test_Dataset
from metric import calculate_psnr, calculate_ssim, calculate_psnr_np
from utils import AverageMeter
import logging
from skimage.metrics import structural_similarity as ssim


def get_lr(args, iters):
    ratio = 0.5 * (1.0 + np.cos(iters / (args.epochs * args.iters_per_epoch) * math.pi))
    lr = (args.lr_start - args.lr_end) * ratio + args.lr_end
    return lr


def set_lr(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def train(args, model):
    os.makedirs(args.log_path, exist_ok=True)
    log_path = os.path.join(args.log_path, time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()))
    os.makedirs(log_path, exist_ok=True)
    logger = logging.getLogger()
    logger.setLevel('INFO')
    BASIC_FORMAT = '%(asctime)s:%(levelname)s:%(message)s'
    DATE_FORMAT = '%Y-%m-%d %H:%M:%S'
    formatter = logging.Formatter(BASIC_FORMAT, DATE_FORMAT)
    chlr = logging.StreamHandler()
    chlr.setFormatter(formatter)
    chlr.setLevel('INFO')
    fhlr = logging.FileHandler(os.path.join(log_path, 'train.log'))
    fhlr.setFormatter(formatter)
    logger.addHandler(chlr)
    logger.addHandler(fhlr)
    logger.info(args)

    dataset_train = Pre_Train_Dataset(dataset_dir='/home/Videos/train', augment=True)
    dataloader_train = DataLoader(dataset_train, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True, drop_last=True) 
    args.iters_per_epoch = dataloader_train.__len__()
    iters = args.resume_epoch * args.iters_per_epoch
    
    dataset_val = Pre_Test_Dataset(dataset_dir='/home/Videos/test')
    dataloader_val = DataLoader(dataset_val, batch_size=1, num_workers=1, pin_memory=True, shuffle=False, drop_last=True)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr_start, weight_decay=0)

    time_stamp = time.time()
    avg_rec = AverageMeter()
    avg_geo = AverageMeter()
    avg_dis = AverageMeter()
    best_psnr = 0.0

    for epoch in range(args.resume_epoch, args.epochs):
        for i, data in enumerate(dataloader_train):
            for l in range(len(data)):
                data[l] = data[l].to(args.device)

            # with torch.autograd.set_detect_anomaly(True):
            img0, imgt, img1, flow, embt = data

            data_time_interval = time.time() - time_stamp
            time_stamp = time.time()

            lr = get_lr(args, iters)
            set_lr(optimizer, lr)

            optimizer.zero_grad()

            imgt_pred, loss_rec, loss_geo, loss_dis = model(img0, img1, embt, imgt, flow)

            loss = loss_rec + loss_geo + loss_dis
            loss.backward()
            optimizer.step()

            avg_rec.update(loss_rec.cpu().data)
            avg_geo.update(loss_geo.cpu().data)
            avg_dis.update(loss_dis.cpu().data)
            train_time_interval = time.time() - time_stamp

            if (iters+1) % (args.iters_per_epoch) == 0:
                logger.info('epoch:{}/{} iter:{}/{} time:{:.2f}+{:.2f} lr:{:.5e} loss_rec:{:.4e} loss_geo:{:.4e} loss_dis:{:.4e}'.format(epoch+1, args.epochs, iters+1, args.epochs * args.iters_per_epoch, data_time_interval, train_time_interval, lr, avg_rec.avg, avg_geo.avg, avg_dis.avg))
                avg_rec.reset()
                avg_geo.reset()
                avg_dis.reset()

            iters += 1
            time_stamp = time.time()
        
        torch.save(model.state_dict(), '{}/{}_{}.pth'.format(log_path, args.model_name, 'latest'))     
        if (epoch+1) % args.eval_interval == 0:
            psnr = evaluate(args, model, dataloader_val, epoch, logger)
            if psnr > best_psnr:
                best_psnr = psnr
                torch.save(model.state_dict(), '{}/{}_{}.pth'.format(log_path, args.model_name, 'best'))
            # torch.save(model.module.state_dict(), '{}/{}_{}.pth'.format(log_path, args.model_name, 'latest'))
            
            torch.save(model.state_dict(), '{}/{}_{}.pth'.format(log_path, args.model_name, "%03d"%(epoch+1)))


def evaluate(args, model, dataloader_val, epoch, logger):
    loss_rec_list = []
    loss_geo_list = []
    loss_dis_list = []
    psnr_list = []
    ssim_list = []
    time_stamp = time.time()
    for i, data in enumerate(dataloader_val):
        for l in range(len(data)):
            data[l] = data[l].squeeze(0).to(args.device)
        img0, imgt, img1, flow, embt = data

        with torch.no_grad():
            imgt_pred, loss_rec, loss_geo, loss_dis = model(img0, img1, embt, imgt, flow)

        loss_rec_list.append(loss_rec.cpu().numpy())
        loss_geo_list.append(loss_geo.cpu().numpy())
        loss_dis_list.append(loss_dis.cpu().numpy())

        for j in range(img0.shape[0]):
            img_pred_np = (imgt_pred[j].data.permute(1, 2, 0).cpu().numpy() * 255.0).astype(np.uint8)
            imgt = (imgt[j].data.permute(1, 2, 0).cpu().numpy() * 255.0).astype(np.uint8)
            psnr = calculate_psnr_np(imgt, img_pred_np)
            ssim_value=ssim(imgt, img_pred_np,multichannel=True)
            psnr_list.append(psnr)
            ssim_list.append(ssim_value)

    eval_time_interval = time.time() - time_stamp
    
    logger.info('eval epoch:{}/{} time:{:.2f} loss_rec:{:.4e} loss_geo:{:.4e} loss_dis:{:.4e} psnr:{:.3f} ssim:{:.3f}'.format(epoch+1, args.epochs, eval_time_interval, np.array(loss_rec_list).mean(), np.array(loss_geo_list).mean(), np.array(loss_dis_list).mean(), np.array(psnr_list).mean(), np.array(ssim_list).mean()))
    return np.array(psnr_list).mean()



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='IFRNet')
    parser.add_argument('--model_name', default='IFRNet_S', type=str, help='IFRNet, IFRNet_L, IFRNet_S')
    parser.add_argument('--epochs', default=1000, type=int)
    parser.add_argument('--eval_interval', default=10, type=int)
    parser.add_argument('--batch_size', default=18, type=int)
    parser.add_argument('--lr_start', default=1e-4, type=float)
    parser.add_argument('--lr_end', default=1e-5, type=float)
    parser.add_argument('--log_path', default='checkpoint', type=str)
    parser.add_argument('--resume_epoch', default=0, type=int)
    parser.add_argument('--resume_path', default=None, type=str)
    args = parser.parse_args()

    args.device = torch.device('cuda')

    seed = 1234
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True

    if args.model_name == 'IFRNet':
        from models.IFRNet import Model
    elif args.model_name == 'IFRNet_L':
        from models.IFRNet_L import Model
    elif args.model_name == 'IFRNet_S':
        from models.IFRNet_S_copy_2 import Model

    args.log_path = args.log_path + '/' + args.model_name
    args.num_workers = args.batch_size

    model = Model().to(args.device)
    
    if args.resume_epoch != 0:
        model.load_state_dict(torch.load(args.resume_path, map_location='cpu'))
        
    
    train(args, model)

