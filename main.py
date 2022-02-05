import os
import torch
import logging
import argparse
from torch import autograd, optim
from torch.utils.data import DataLoader
from UNet import UNet, ResNet34_UNet
from torchvision.transforms import transforms

from dataset import *
from metrics import *

def getArgs():
    parse = argparse.ArgumentParser()
    parse.add_argument('--action', type=str, help="Choose between train, test or borth training and testing loops", default="train&test")
    parse.add_argument('--epoch', type=int, default=41)
    parse.add_argument('--arch', '-a', metavar='ARCH',
            default='ResNet34_UNet',
            help="Choose the segmentation architecture from: UNet/ResNet34_UNet")
    parse.add_argument('--batch_size', type=int, default=1)
    parse.add_argument('--dataset', default='',
            help = "Choose from the different datasets")
    parse.add_argument('--log-dir', default='result/log',
            help="The logs directory location")
    parse.add_argument('--threshold', type=float, default=None)
    args = parse.parse_args()
    return args

def getLog(args):
    dirname = os.path.join(args.log_dir, args.arch, str(args.batch_size),
            str(args.dataset), str(args.epoch))
    filename = dirname + '/log.log'
    if not os.path.exists(dirname):
        os.makedirs(dirname)

    logging.basicConfig(
            filename=filename,
            level=logging.DEBUG,
            format='%(asctime)s:%(levelname)s:%(message)s'
            )
    return logging

def getModel(args):
    if args.arch == 'UNet':
        model = UNet(3,1).to(device)
    if args.arch == 'ResNet34_UNet':
        model = ResNet34_UNet(1, pretrained=False).to(device)

def getDataset(args):
    train_dataloaders, val_dataloaders, test_dataloaders = None, None, None
    if args.dataset=='remote-sensing':
        train_dataset = RemoteSensing('train', transform=x_transforms, 
                target_transform=y_transforms)
        train_dataloaders = DataLoader(train_dataset, batch_size=args.batch_size)
        val_dataset = RemoteSensing('val', transform=x_transforms,
                target_transform=y_transforms)
        val_dataloaders = DataLoader(val_dataset, batch_size=1)
        test_dataloaders = val_dataloaders

    return train_dataloaders, val_dataloaders, test_dataloaders

def val(model, best_iou, val_dataloaders):
    model = model.eval()
    with torch.no_grad():
        i = 0
        miou_total = 0
        hd_total = 0
        dice_total = 0
        num = len(val_dataloaders)

        for x, _, pic, mask in val_dataloaders:
            x = x.to(device)
            y = model(x)

            if args.deepsupervision:
                img_y = torch.squeeze(y[-1]).cpu().numpy()
            else:
                img_y = torch.squeeze(y).cpu().numpy()

            hd_total += get_hd(mask[0], img_y)
            miou_total += get_iou(mask[0], img_y)
            dice_total += get_dice(mask[0], img_y)
            if i < num: i+=1

        aver_iou = miou_total / num
        aver_hd = hd_total / num
        aver_dice = dice_total / num

        print(f"MIOU {aver_iou}   AVER_HD {aver_hd}   AVER_DICE {aver_dice}")
        
        if aver_iou > best_iou:
            print(f"AVER_IOU {aver_iou} > BEST_IOU {best_iou}")
            best_iou = aver_iou
            print("[INFO] Saving best model...")
            torch.save(model.state_dict(), f"./saved_model/{args.arch}_{args.batch_size}_{args.dataset}_{args.epoch}.pth")

        return best_iou, aver_iou, aver_dice, aver_hd


def train(model, criterion, optimizer, train_dataloader, val_dataloader, args):
    pass

def test(val_dataloarders, save_predict=False):
    pass

if __name__ == "__main__":
    x_transforms = transforms.ToTensor()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(device)

    args = getArgs()
    logging = getLog(args)

    print('='*25)
    print(f"models:{args.arch},\nepoch:{args.epoch},\nbatch size:{args.batch_size}\ndataset:{args.dataset}")
    logging.info(f"\n{25*'-'}\nmodels:{args.arch},\nepoch:{args.epoch},\nbatch size:{args.batch_size}\ndataset:{args.dataset}\n{25*'-'}\n")
    print('='*25)

    model = getModel(args)

    train_dataloaders, val_dataloaders, test_dataloaders = getDataset(args)

    criterion = torch.nn.BCELoss()
    optimizer = optim.Adam(model.parameters())

    if 'train' in args.action:
        train(model, criterion, optimizer, train_dataloaders, val_dataloaders, args)
    if 'test' in args.action:
        test(test_dataloaders, save_predict=True)

